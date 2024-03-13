# basic
import os.path as osp

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from active_selection.utils import BaseSelector, get_al_loader


def kcenter_greedy(dist_mat, n_data, budget, init_idx):
    assert dist_mat.shape[0] == n_data, (
        "Size of distance matrix and number of data doesn't match!")

    # init
    all_indices = np.arange(n_data)
    labeled_indices = np.zeros((n_data, ), dtype=np.bool8)
    labeled_indices[init_idx] = True

    # sample 
    for _ in tqdm(range(budget), desc="k-center greedy"):
        mat = dist_mat[~labeled_indices, :][:, labeled_indices]
        # for all the unselected points, find its nearest neighbor in selected points
        mat_min = mat.min(axis=1)
        # find nearest neighbor with largest distance as the next selected point
        q_index_ = mat_min.argmax()
        q_index = all_indices[~labeled_indices][q_index_]
        labeled_indices[q_index] = True
    
    selected_idx = all_indices[labeled_indices]
    all_else_idx = all_indices[~labeled_indices]
    newly_selected_idx = list(set(selected_idx) - set(init_idx))

    return newly_selected_idx


class CoreSetSelector(BaseSelector):
    def __init__(self, args, logger, batch_size, num_workers):
        super().__init__(args, logger, batch_size, num_workers)

    def calculate_scores(self, trainer, active_set):
        model = trainer.net
        model.eval()

        # concat ALL datasets
        label_dataset = active_set.label_dataset
        pool_dataset = active_set.pool_dataset
        core_list = label_dataset.im_idx
        all_list = core_list + pool_dataset.im_idx
        all_dataset = ConcatDataset([label_dataset, pool_dataset])
        loader = get_al_loader(all_dataset, self.batch_size, self.num_workers)

        # get encoder feature for every sample
        feature = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader, desc="core_set feat")):
                # get data
                images = batch['images']
                images = images.to(trainer.device, dtype=torch.float32)

                # forward
                if trainer.data_parallel:
                    feats = model.module.get_enc_feature(images).cpu().numpy()      # [B, C]
                else:
                    feats = model.get_enc_feature(images).cpu().numpy()             # [B, C]
                feature.append(feats)

        feature = np.concatenate(feature, axis=0)                                       # [N, C]
        feat_dist_mat = pairwise_distances(feature, metric=self.args.coreset_distance)  # [N, N]

        return np.array(core_list), np.array(all_list), feature, feat_dist_mat


    def select_next_batch(self, trainer, active_set, selection_count, selection_iter):
        # get list, feature and feature distance matrix
        core_list, all_list, feature, feat_dist_mat = self.calculate_scores(trainer, active_set)

        # k-center greedy sampling
        newly_selected_idx = kcenter_greedy(
            dist_mat=feat_dist_mat, 
            n_data=len(all_list), 
            budget=selection_count, 
            init_idx=np.arange(len(core_list))
        )

        # expand training set
        active_set.expand_training_set(all_list[newly_selected_idx].tolist())

        # save feature and feature distance matrix
        save_path = osp.join(
            trainer.exp_dir, "record", f"coreset_round{selection_iter}.npz")
        np.savez_compressed(save_path, feat=feature, dist_mat=feat_dist_mat)
