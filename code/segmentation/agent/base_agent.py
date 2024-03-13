import time
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medpy.metric import dc, asd
from monai.losses import DiceLoss
from monai.networks import one_hot
from torch.optim.lr_scheduler import LambdaLR

from models import get_model
from dataloader.utils import DataProvider
from utils.common import AverageMeter, ProgressMeter


class BaseTrainer(object):
    def __init__(self, args, logger, writer, selection_iter, print_freq=10):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.print_freq = print_freq
        self.selection_iter = selection_iter      # -1: warmup for active learning

        self.best_dice = 0.                     # mean dice
        self.exp_dir = args.exp_dir
        total_itrs = self.args.total_itrs
        self.num_classes = args.num_classes
        self.data_parallel = args.data_parallel
        self.device = torch.device(args.device)

        # model
        self.net = self._model_wrapper(get_model(args))
        self.logger.info(f"==> Model Architecture...")
        self.logger.info(self.net)

        # loss
        self.criterion_dice = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

        # optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), args.lr, weight_decay=args.weight_decay)
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.net.parameters(), args.lr, weight_decay=args.weight_decay)
        elif self.args.optimizer == 'sgdm':
            self.optimizer = optim.SGD(self.net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        # learning rate schedule
        if self.args.lr_schedule == 'poly':
            self.scheduler = LambdaLR(self.optimizer, lambda itr: (1 - itr / total_itrs) ** 0.9)
        elif self.args.lr_schedule == 'none':
            self.scheduler = None

    def _model_wrapper(self, net):
        if self.args.data_parallel:
            net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
        else:
            net = net.to(self.device)
        return net

    def get_trainloader(self, dataset, total_iters=None):
        data_provider = DataProvider(dataset=dataset, batch_size=self.args.train_batch_size,
                                     shuffle=True, num_workers=self.args.num_workers, 
                                     pin_memory=True, drop_last=False, total_iters=total_iters)
        return data_provider

    def get_inferloader(self, dataset):

        data_provider = DataProvider(dataset=dataset, batch_size=self.args.val_batch_size, 
                                     shuffle=False, num_workers=self.args.num_workers,
                                     pin_memory=True, drop_last=False)
        return data_provider

    def train_impl(self, total_itrs, val_period):
        self.net.train()

        # define meters for logging
        data_time = AverageMeter('Data', ':6.3f')
        batch_time = AverageMeter('Time', ':6.3f')
        loss_ce_meter = AverageMeter('CE', ':.4f')
        loss_dice_meter = AverageMeter('Dice', ':.4f')
        loss_meter = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            total_itrs, [batch_time, data_time, loss_ce_meter, loss_dice_meter, loss_meter], 
            prefix=f"Train: ")

        # train loop
        end = time.time()
        for iteration in range(total_itrs):
            # get data
            batch = self.train_dataset_loader.__next__()
            images = batch['images']
            labels = batch['labels']
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            bsz = images.size(0)
            data_time.update(time.time() - end)

            # forward
            outputs = self.net(images)    # (B, C, H, W)
            # updates the gradient for unmasked area, works for both image and region mode
            loss_mask = (labels != self.args.ignore_index)
            loss_dice = self.criterion_dice(outputs, labels, mask=loss_mask)
            loss_ce = self.criterion_ce(outputs, labels.squeeze(1))
            loss = loss_dice + loss_ce

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=self.args.grad_norm)  # clip gradients
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            torch.cuda.synchronize()
            loss_ce_meter.update(loss_ce.item(), bsz)
            loss_dice_meter.update(loss_dice.item(), bsz)
            loss_meter.update(loss.item(), bsz)
            batch_time.update(time.time() - end)

            # tensorboard
            tag_suffix = '' if self.selection_iter < 0 else f"_round{self.selection_iter}"
            self.writer.add_scalar("train/loss" + tag_suffix, loss.item(), iteration)
            self.writer.add_scalar("train/loss_ce" + tag_suffix, loss_ce.item(), iteration)
            self.writer.add_scalar("train/loss_dice" + tag_suffix, loss_dice.item(), iteration)
            self.writer.add_scalar("train/lr" + tag_suffix, 
                self.optimizer.state_dict()['param_groups'][0]['lr'], iteration)
            self.writer.add_scalar("train/grad_norm" + tag_suffix, grad_norm.item(), iteration)

            # terminal and logger display
            if ((iteration == 0) or 
                ((iteration + 1) % self.print_freq == 0) or 
                ((iteration + 1) == total_itrs)):
                progress.display(iteration + 1, self.logger)

            # validation
            if (iteration + 1) % val_period == 0:
                self.logger.info(f"==> Validation at Iteration {iteration}...")
                self.validate(self.val_dataset_loader, "val", step=iteration)
            
            end = time.time()

    def validate(self, dataloader, mode:str, step:int, update_ckpt=True):
        """
            loader: loader of validation set or test set 
            mode: "validate", "infer, "active"
            step: could be #train_iter or #selection_iter 
        """
        self.net.eval()
        N = dataloader.__len__()

        data_time = AverageMeter('Data', ':6.3f')
        batch_time = AverageMeter('Time', ':6.3f')
        mean_dice_meter = AverageMeter('mean_Dice', ':6.3f')
        mean_asd_meter = AverageMeter('mean_ASD', ':6.3f')
        dice_meters = [AverageMeter(f'Dice_{i+1}', ':6.3f') for i in range(self.num_classes - 1)]
        asd_meters = [AverageMeter(f'ASD_{i+1}', ':6.3f') for i in range(self.num_classes - 1)]
        progress = ProgressMeter(
            N, [batch_time, data_time, mean_dice_meter, mean_asd_meter], 
            prefix=f"{mode.capitalize()}: ")

        with torch.no_grad():
            end = time.time()
            for iteration in range(N):
                # get data
                batch = dataloader.__next__()
                images = batch['images']    # (B, 1, H, W) or (1, 1, D, H', W')
                labels = batch['labels']    # (B, 1, H, W) or (1, 1, D, H, W)
                spacing = batch['spacing'].cpu().numpy()  if 'spacing' in batch else None
                names = np.array(batch['fnames'])[:, 0]
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                data_time.update(time.time() - end)

                if self.args.val_mode == 'slice':
                    self.infer_slice(images, labels, dice_meters, asd_meters, spacing=spacing)
                elif self.args.val_mode == 'volume':
                    self.infer_volume(images, labels, dice_meters, asd_meters, spacing=spacing)

                # logging
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)
                mean_dice_meter.update(np.mean([meter.avg for meter in dice_meters]))
                mean_asd_meter.update(np.mean([meter.avg for meter in asd_meters]))

                # terminal and logger display
                progress.display(iteration + 1, self.logger)

                end = time.time()
            
            # tensorboard
            mean_dice = np.mean([meter.avg for meter in dice_meters])
            mean_asd = np.mean([meter.avg for meter in asd_meters])
            self.writer.add_scalar(f"{mode}/mean_dice", mean_dice, step)
            self.writer.add_scalar(f"{mode}/mean_asd", mean_asd, step)
            for i, (dice_meter, asd_meter) in enumerate(zip(dice_meters, asd_meters)):
                self.writer.add_scalar(f"{mode}/dice_{i+1}", dice_meter.avg, step)
                self.writer.add_scalar(f"{mode}/asd_{i+1}", asd_meter.avg, step)

            # metrics str & metrics df logging
            dice_header = [f"dice_{i+1}" for i in range(len(dice_meters))] + ['mean_dice']
            dice_data = [meter.avg for meter in dice_meters] + [mean_dice]
            asd_header = [f"asd_{i+1}" for i in range(len(asd_meters))] + ['mean_asd']
            asd_data = [meter.avg for meter in asd_meters] + [mean_asd]
            header = dice_header + asd_header
            metrics = dice_data + asd_data
            metrics_df = pd.DataFrame({s:d for s,d in zip(header, metrics)}, index=[step])
            metrics_str = " ".join([f"{s},{d:.3f}" for s,d in zip(header, metrics)])
            self.logger.info('[Results]: ')
            self.logger.info(f'{metrics_str}')

            # save best model            
            if update_ckpt is False:
                return metrics_df
            if self.best_dice < mean_dice:
                self.best_dice = mean_dice
                self.save_checkpoint()

        # switch back to train mode
        self.net.train()

        return metrics_df

    def infer_slice(self, images, labels, dice_meters, asd_meters, spacing=None):
        # forward
        outputs = self.net(images)                              # (B, C, H, W)
        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = F.interpolate(
                outputs, size=labels.shape[-2:], mode="area")
        probs = torch.softmax(outputs, dim=1)                   # (B, C, H, W)
        preds = torch.argmax(probs, dim=1, keepdim=True)        # (B, 1, H, W) 
        
        # to one-hot
        B, C = outputs.shape[:2]
        preds_onehot = one_hot(preds, num_classes=C, dim=1).cpu().numpy()     # (B, C, H, W)
        labels_onehot = one_hot(labels, num_classes=C, dim=1).cpu().numpy()   # (B, C, H, W)

        # update dice meter & skip empty class
        dice = np.zeros((B, C-1))
        for b in range(B):
            for c in range(1, C):
                if labels_onehot[b, c].any():
                    dice_bc = dc(preds_onehot[b, c], labels_onehot[b, c])
                    dice_meters[c-1].update(dice_bc)
                    dice[b, c-1] = dice_bc
        
        # update asd meter: both label and pred should have binary object
        asdis = np.zeros((B, C-1))
        for b in range(B):
            for c in range(1, C):
                if preds_onehot[b, c].any() and labels_onehot[b, c].any():
                    asd_bc = asd(preds_onehot[b, c], labels_onehot[b, c], 
                                 voxelspacing=spacing[b, :2])
                    asd_meters[c-1].update(asd_bc)
                    asdis[b, c-1] = asd_bc

    def infer_volume(self, images, labels, dice_meters, asd_meters, spacing=None):
        """
            images: [1, 1, D, IMG_SIZE, IMG_SIZE]
            labels: [1, 1, D, H, W]
            IMG_SIZE usually equals to 256
        """
        assert self.args.val_batch_size == 1, "batch size of volume infer must be 1"

        # turn 3D input to batched 2D input
        IMG_SIZE, C = self.args.img_size, self.args.num_classes
        D, H, W = labels.shape[2:]
        images = images.view(D, 1, IMG_SIZE, IMG_SIZE)              # (D, 1, IMG_SIZE, IMG_SIZE) 
        labels = labels.view(1, D, H, W)                            # (1, D, H, W) 

        # forward, resize probability to original size
        outputs = self.net(images)                                  # (D, C, IMG_SIZE, IMG_SIZE)
        probs = torch.softmax(outputs, dim=1)                       # (D, C, IMG_SIZE, IMG_SIZE)
        probs = F.interpolate(probs, size=(H, W), mode="area")      # (D, C, H, W)
        preds = torch.argmax(probs, dim=1, keepdim=True)            # (D, 1, H, W)
        preds = preds.view(1, D, H, W)                              # (1, D, H, W)

        # to one-hot
        preds_onehot = one_hot(preds, num_classes=C, dim=0).cpu().numpy()
        labels_onehot = one_hot(labels, num_classes=C, dim=0).cpu().numpy()

        # update dice meter & skip empty class
        for c in range(1, C):
            if labels_onehot[c-1].any():
                dice_c = dc(preds_onehot[c], labels_onehot[c])
                dice_meters[c-1].update(dice_c)
        
        # update asd meter: both label and pred should have binary object
        for c in range(1, C):
            if preds_onehot[c].any() and labels_onehot[c].any():
                voxelspacing = np.roll(spacing[0], shift=1)     # spacing of [x, y, z] -> [z, x, y]
                asd_bc = asd(preds_onehot[c], labels_onehot[c], voxelspacing=voxelspacing)
                # asd_bc = asd(preds_onehot[c], labels_onehot[c])
                asd_meters[c-1].update(asd_bc)

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'opt_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self, fname, load_optimizer=False):
        map_location = self.device
        checkpoint = torch.load(fname, map_location=map_location)
        model_state_dict = checkpoint['model_state_dict']
        dp_key = [k.startswith("module.") for k in model_state_dict.keys()]

        # use data parallel and model is not wrapped with DataParallel
        if self.args.data_parallel and not any(dp_key):
            dp_state_dict = OrderedDict({('module.' + k): v for k, v in model_state_dict.items()})
            self.net.load_state_dict(dp_state_dict)
        else:
            self.net.load_state_dict(model_state_dict)

        # load optimizer (usually don't)
        if load_optimizer is True:
            self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
