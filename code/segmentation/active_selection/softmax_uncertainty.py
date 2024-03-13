import os.path as osp

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


class SoftmaxUncertaintySelector(BaseSelector):
    def __init__(self, args, logger, batch_size, num_workers, active_method):
        super().__init__(args, logger, batch_size, num_workers)
        self.active_method = active_method
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']
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
