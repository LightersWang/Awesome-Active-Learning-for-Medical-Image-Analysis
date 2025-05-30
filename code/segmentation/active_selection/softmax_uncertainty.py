import os.path as osp
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from active_selection.utils import BaseSelector, get_al_loader

"""
- Image-based Active Learning Example: SoftmaxUncertaintySelector
    - Key idea: choose the minimal max_prob image as the most valuable one.
    - select_next_batch --> calculate_scores --> active_set.expand_training_set()
    - calculate_scores: inference all poolset and then record their scores.
"""


# All larger the better

def least_confidence(outputs, softmax=True):
    probs = F.softmax(outputs, dim=1) if softmax else outputs
    CONF = torch.max(probs, 1)[0]
    CONF *= -1  # The small the better --> Reverse it makes it the large the better
    return CONF


def least_margin(outputs, softmax=True):
    probs = F.softmax(outputs, dim=1) if softmax else outputs
    TOP2 = torch.topk(probs, 2, dim=1)[0]
    MARGIN = TOP2[:, 0] - TOP2[:, 1]
    MARGIN *= -1   # The small the better --> Reverse it makes it the large the better
    return MARGIN


def max_entropy(outputs, softmax=True):
    # Softmax Entropy
    probs = F.softmax(outputs, dim=1) if softmax else outputs
    ENT = torch.mean(-probs * torch.log2(probs + 1e-12), dim=1)  # The large the better
    return ENT


def max_kl_div(ens_probs):      # (N, B, C, H, W)
    mean_prob = ens_probs.mean(0)
    kl = []
    for i in range(ens_probs.shape[0]):
        kl.append(F.kl_div(ens_probs[i].log(), mean_prob, reduction='none'))
    kl_div = torch.stack(kl, dim=0).mean([0, 2])
    return kl_div           # (B, H, W)


def max_bald(mcdropout_probs, mcdropout_probs_mean):
    entropy_mean_prob = -torch.mean(
        mcdropout_probs_mean * torch.log2(mcdropout_probs_mean + 1e-12), dim=1)
    
    entropy_probs = []
    for prob in mcdropout_probs:
        entropy_prob = -torch.mean(prob * torch.log2(prob + 1e-12), dim=1)
        entropy_probs.append(entropy_prob)
    entropy_probs = torch.stack(entropy_probs, dim=0)
    mean_entropy = entropy_probs.mean(0)
    bald = entropy_mean_prob - mean_entropy
    return bald


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class SoftmaxUncertaintySelector(BaseSelector):
    def __init__(self, args, logger, batch_size, num_workers, active_method):
        super().__init__(args, logger, batch_size, num_workers)
        self.active_method = active_method
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy', 'bald']
        if active_method == 'softmax_confidence':
            self.uncertain_handler = least_confidence
        if active_method == 'softmax_margin':
            self.uncertain_handler = least_margin
        if active_method == 'softmax_entropy':
            self.uncertain_handler = max_entropy

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()

        loader = get_al_loader(pool_set, self.batch_size, self.num_workers)
        tqdm_loader = tqdm(loader, total=len(loader), desc=self.active_method)

        all_scores, all_fnames = [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm_loader):
                # get data
                images = batch['images']
                fnames = batch['fnames']
                images = images.to(trainer.device, dtype=torch.float32)

                # uncertainty score for every sample
                torch.cuda.synchronize()
                outputs = model(images)                                     # (B, C, H, W)
                uncertainty = self.uncertain_handler(outputs, softmax=True) # (B, H, W)
                scores = uncertainty.mean(dim=[1, 2]).cpu().numpy()         # (B,)

                # update scores and fnames
                all_scores.extend(scores)
                all_fnames.extend(fnames)
        
        # save scores in csv
        all_img_names = np.array(all_fnames, dtype=str).squeeze()
        scores_df = pd.DataFrame(
            {'img_path': all_img_names, 'score': all_scores})
        
        # descending order, larger the better
        scores_df = scores_df.sort_values(by='score', ascending=False)      

        return scores_df

    def select_next_batch(self, trainer, active_set, selection_count, selection_iter):
        # get tables with fnames and scores
        scores_df = self.calculate_scores(trainer, active_set.pool_dataset)

        # select largest #selection_count samples
        # samples == [img_path + spx_path]
        selected_samples = scores_df.iloc[:selection_count, :1].to_numpy().tolist()
        active_set.expand_training_set(selected_samples)

        # save scores df
        scores_path = osp.join(
            trainer.exp_dir, "record", f"{self.active_method}_round{selection_iter}.csv")
        scores_df.to_csv(scores_path)

class MCDropoutSelector(SoftmaxUncertaintySelector):
    def __init__(self, args, logger, batch_size, num_workers, active_method, repeat_times=3):
        super().__init__(args, logger, batch_size, num_workers, active_method)
        self.repeat_times = repeat_times
        assert active_method in ['softmax_entropy', 'bald']
        if active_method == 'softmax_entropy':
            self.uncertain_handler = max_entropy
        elif active_method == 'bald':
            pass
        else:
            raise NotImplementedError(f"{active_method} is not supported for MCDropoutSelector")

    def calculate_scores(self, trainer, pool_set):
        # enables dropout layer only in decoder
        model = deepcopy(trainer.net)
        model.eval()
        enable_dropout(model)

        loader = get_al_loader(pool_set, self.batch_size, self.num_workers)
        tqdm_loader = tqdm(loader, total=len(loader), desc=f"mcdropout_{self.active_method}")

        all_scores, all_fnames = [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm_loader):
                # get data
                images = batch['images']
                fnames = batch['fnames'] 
                images = images.to(trainer.device, dtype=torch.float32)

                # multiple forward passes with dropout enabled
                torch.cuda.synchronize()
                mcdropout_probs = []
                for _ in range(self.repeat_times):
                    mcdropout_probs.append(F.softmax(model(images), dim=1))    # (B, C, H, W)
                
                # uncertainty after mc dropout
                mcdropout_probs = torch.stack(mcdropout_probs, dim=0)          # (N, B, C, H, W)
                mcdropout_probs_mean = mcdropout_probs.mean(dim=0)            # (B, C, H, W)
                if self.active_method == 'meanstd':
                    uncertainty = mcdropout_probs.std(dim=0).mean(dim=1)      # (B, H, W)
                elif self.active_method == 'bald':
                    uncertainty = max_bald(
                        mcdropout_probs, mcdropout_probs_mean)
                else:
                    uncertainty = self.uncertain_handler(
                        mcdropout_probs_mean, softmax=False)                 # (B, H, W)
                scores = uncertainty.mean(dim=[1, 2]).cpu().numpy()          # (B,)

                # update scores and fnames 
                all_scores.extend(scores)
                all_fnames.extend(fnames)

        # save scores in csv
        all_img_names = np.array(all_fnames, dtype=str).squeeze()
        scores_df = pd.DataFrame(
            {'img_path': all_img_names, 'score': all_scores})
        
        # descending order, larger the better
        scores_df = scores_df.sort_values(by='score', ascending=False)      

        return scores_df
