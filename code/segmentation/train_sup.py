#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
from datetime import datetime

from agent.base_agent import BaseTrainer
from dataloader import get_dataset
from utils.common import initialization, timediff
from utils.configs import get_args


class SupTrainer(BaseTrainer):
    def __init__(self, args, logger, writer):
        super().__init__(args, logger, writer, selection_iter=-1)

    def train(self):
        self.logger.info("==> Training starts (Supervised Trainer)...")

        # datasets
        train_set = get_dataset(self.args, split=self.args.train_split)
        val_dataset = get_dataset(self.args, split=self.args.val_split)
        test_dataset = get_dataset(self.args, split=self.args.test_split)

        self.logger.info(f"Number of Train Set:\t{len(train_set)}")
        self.logger.info(f"Number of Val Set:\t{len(val_dataset)}")
        self.logger.info(f"Number of Test Set:\t{len(test_dataset)}")

        # dataloaders
        self.train_dataset_loader = self.get_trainloader(train_set)
        self.val_dataset_loader = self.get_inferloader(val_dataset)
        self.test_dataset_loader = self.get_inferloader(test_dataset)
        
        self.checkpoint_file = osp.join(self.exp_dir, 'checkpoint.tar')

        total_itrs = int(self.args.total_itrs)
        val_period = int(self.args.val_period)
        self.train_impl(total_itrs, val_period)


def main(args):
    # initialization
    logger, writer = initialization(args)
    t_start = datetime.now()

    # Training
    trainer = SupTrainer(args, logger, writer)
    trainer.train()

    # Evaluate on Test Set 
    fname = osp.join(args.exp_dir, 'checkpoint.tar')
    trainer.load_checkpoint(fname)
    test_result = trainer.validate(
        trainer.test_dataset_loader, mode='test', step=0, update_ckpt=False)
    t_end = datetime.now()

    # End Experiment
    t_end = datetime.now()
    logger.info(f"{'%'*20} Experiment Report {'%'*20}")
    logger.info(f"0. Methods: Fully Supervision")
    logger.info(f"1. Takes: {timediff(t_start, t_end)}")
    logger.info(f"2. Log dir: {args.exp_dir} (with model checkpoint)")
    logger.info(f"3. Test Metircs")
    logger.info(test_result)
    logger.info(f"{'%'*20} Experiment End {'%'*20}")


if __name__ == '__main__':
    args = get_args(mode='supervised')
    
    print(' '.join(sys.argv))
    print(args)
    main(args)
