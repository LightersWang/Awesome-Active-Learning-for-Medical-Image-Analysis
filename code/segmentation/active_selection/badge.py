import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks import one_hot
from sklearn.cluster import kmeans_plusplus
from torch_scatter import scatter_mean
from tqdm import tqdm

from active_selection.utils import BaseSelector, get_al_loader


def image_wise_grad(args, loss, model, last_layer_name="decoder.seg_output.weight"):
    """
        grad_embeddimg: a tensor of shape [C*D,] for one sample
    """

    if args.data_parallel:
        last_layer_name = "module." + last_layer_name

    model.zero_grad()
    loss.backward(retain_graph=True)
    last_layer_param = dict(model.named_parameters())[last_layer_name]
    last_layer_grad = last_layer_param.grad
    grad_embeddimg = last_layer_grad.detach().flatten().clone()     # clone is important!
    
    return grad_embeddimg


class BADGESelector(BaseSelector):
    def __init__(self, args, logger, batch_size, num_workers, multiple_loss='add'):
        # batch size must be 1 for gradient calculation
        super().__init__(args, logger, batch_size=1, num_workers=num_workers)
        self.multiple_loss = multiple_loss

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()

        loader = get_al_loader(pool_set, self.batch_size, self.num_workers)
        tqdm_loader = tqdm(loader, total=len(loader), desc="Image BADGE")

        all_grad_embed = []
        for _, batch in enumerate(tqdm_loader):
            # get data
            images = batch['images']
            images = images.to(trainer.device, dtype=torch.float32)

            # preds as pseudo label to calculate loss
            torch.cuda.synchronize()
            outputs = model(images)                                         # [1, C, H, W]
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)          # [1, H, W]

            if self.multiple_loss == 'sep':
                # cross entropy loss gradient embedding
                criterion_ce = trainer.criterion_ce
                loss_ce = criterion_ce(outputs, preds)
                grad_embed_ce = image_wise_grad(self.args, loss_ce, model)    # [C*D, ]

                # dice loss gradinet embedding
                criterion_dice = trainer.criterion_dice
                loss_dice = criterion_dice(outputs, preds.unsqueeze(1))
                grad_embed_dice = image_wise_grad(self.args, loss_dice, model)    # [C*D, ]

                # update gradient embeddings
                grad_embed = torch.cat([grad_embed_ce, grad_embed_dice])    # [2*C*D, ]
            elif self.multiple_loss == 'add':
                # add two losses
                criterion_ce = trainer.criterion_ce
                criterion_dice = trainer.criterion_dice
                loss = criterion_ce(outputs, preds) + criterion_dice(outputs, preds.unsqueeze(1))

                # update gradient embedding
                grad_embed = image_wise_grad(self.args, loss, model)    # [C*D, ]

            all_grad_embed.append(grad_embed)

        pool_list = pool_set.im_idx
        all_grad_embed = torch.stack(all_grad_embed, dim=0).cpu().numpy()    # [N, 2CD]

        return pool_list, all_grad_embed
        

    def select_next_batch(self, trainer, active_set, selection_count, selection_iter):
        # get fnames and gradinet embeddings
        pool_list, all_grad_embed = self.calculate_scores(trainer, active_set.pool_dataset)

        # save gradient embeddings 
        save_path = osp.join(
            trainer.exp_dir, "record", f"badge_round{selection_iter}.npz")
        np.savez(save_path, list=np.array(pool_list), grad_embed=all_grad_embed)

        # diversity sampling via k-means++
        _, selected_indices = kmeans_plusplus(X=all_grad_embed, n_clusters=selection_count)
        selected_samples = np.array(pool_list)[selected_indices].tolist()

        # expand training set with new samples
        active_set.expand_training_set(selected_samples)
