#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import os.path as osp
import sys
import torch
import numpy as np
from shutil import copyfile
from datetime import datetime

# custom
from dataloader import get_dataset
from active_selection import get_active_selector
from agent.base_agent import BaseTrainer
from dataloader import get_active_dataset
from utils.common import active_percent_to_count, finalization, initialization
from utils.configs import get_args


class ALTrainer(BaseTrainer):
    def __init__(self, args, logger, writer, selection_iter):
        super().__init__(args, logger, writer, selection_iter)

    def train(self, active_set):
        self.logger.info("==> Training starts (AL Trainer)...")

        # datasets
        train_set = active_set.get_trainset()
        val_dataset = get_dataset(self.args, split=self.args.val_split)
        test_dataset = get_dataset(self.args, split=self.args.test_split)

        self.logger.info(f"Number of Train Set:\t{len(train_set)}")
        self.logger.info(f"Number of Val Set:\t{len(val_dataset)}")
        self.logger.info(f"Number of Test Set:\t{len(test_dataset)}")

        # dataloaders
        self.train_dataset_loader = self.get_trainloader(train_set, total_iters=self.args.total_itrs)
        self.val_dataset_loader = self.get_inferloader(val_dataset)
        self.test_dataset_loader = self.get_inferloader(test_dataset)

        # checkpoint path
        os.makedirs(osp.join(self.exp_dir, "ckpt"), exist_ok=True)
        self.checkpoint_file = osp.join(
            self.exp_dir, "ckpt", f'checkpoint{active_set.selection_iter:02d}.tar')

        # if the initial model is trained, just copy it
        if (active_set.selection_iter == 0) and (osp.exists(self.args.init_checkpoint)):
            copyfile(self.args.init_checkpoint, self.checkpoint_file)
            self.logger.info(f"==> Load initial model at {args.init_checkpoint}...")
        # if the initial model is not trained or for later rounds, train the model
        else:
            self.train_impl(self.args.total_itrs, self.args.val_period)


def main(args):
    # initialization
    logger, writer = initialization(args)
    t_start = datetime.now()
    val_result, test_result = {}, {}

    # Active Learning setup 
    active_set = get_active_dataset(args)
    selector = get_active_selector(args, logger)
    count_per_round = active_percent_to_count(args, active_set)

    # Active Learning iteration
    logger.info('==> Active Learning Iteration Starts...')
    for selection_iter in range(args.init_iteration, args.max_iterations + 1):
        active_set.selection_iter = selection_iter
        if args.datalist_path is not None:  # resume experiments
            active_set.load_datalist(args.datalist_path)
        
        # initial labeled pool
        if selection_iter == 0:
            args.init_checkpoint = osp.join(
                osp.dirname(osp.dirname(args.exp_dir)), f"init_checkpoint_budget{count_per_round}.tar")
            args.init_datalist = osp.join(
                osp.dirname(osp.dirname(args.exp_dir)), f"init_datalist_budget{count_per_round}.json")
            if not osp.exists(args.init_datalist):
                # not exists, generate initial labled list
                scores = [np.random.random() for _ in range(len(active_set.pool_dataset.im_idx))]
                selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                                    key=lambda x: x[0], reverse=True)))[1][:count_per_round]
                active_set.expand_training_set(selected_samples)
                active_set.dump_initial_datalist(args.init_datalist)
            else:
                # exists, just load it
                active_set.load_initial_datalist(args.init_datalist)

        # 0. Define Trainer
        trainer = ALTrainer(args, logger, writer, selection_iter)

        # 1. Supervision Finetuning
        logger.info(f"==> AL Round {selection_iter} with {count_per_round} ({args.percent_per_round:.2f}%) training data")
        if selection_iter >= 1:     # load ckpt of last iteration
            prevckpt_fname = osp.join(
                args.exp_dir, "ckpt", f'checkpoint{selection_iter-1:02d}.tar')
            trainer.load_checkpoint(prevckpt_fname)
        trainer.train(active_set)

        # 2. Load best checkpoint + Evaluation
        fname = osp.join(args.exp_dir, "ckpt", f'checkpoint{selection_iter:02d}.tar')
        trainer.load_checkpoint(fname)

        # save inital model in iter 0
        if (selection_iter == 0) and (not osp.exists(args.init_checkpoint)):
            copyfile(fname, args.init_checkpoint)
            logger.info(f"==> Save initial model at {args.init_checkpoint}...")

        # validation set
        val_return = trainer.validate(trainer.val_dataset_loader, mode='val', step=selection_iter, update_ckpt=False)
        logger.info(f"==> AL {selection_iter}: Get best validation result")
        val_result[selection_iter] = val_return

        # test set
        test_return = trainer.validate(trainer.test_dataset_loader, mode='test', step=selection_iter, update_ckpt=False)
        logger.info(f"==> AL {selection_iter}: Get test result")
        test_result[selection_iter] = test_return
        torch.cuda.empty_cache()

        # 3. Active Learning selection (skip last iteration)
        if selection_iter < args.max_iterations:
            logger.info(f"==> AL {selection_iter}: Select Next Batch")
            selector.select_next_batch(trainer, active_set, count_per_round, selection_iter)
            active_set.dump_datalist()

    # finalization
    finalization(t_start, val_result, test_result, logger, args)


if __name__ == "__main__":
    # get args
    args = get_args(mode='active_learning')
    print(' '.join(sys.argv))
    print(args)

    main(args)
