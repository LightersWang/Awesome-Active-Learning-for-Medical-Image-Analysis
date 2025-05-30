import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from torch.utils.data.dataset import ConcatDataset
from active_selection.utils import BaseSelector, get_al_loader


def cover_score(cosine_dist, select_indices, all_indices):
    if not np.any(select_indices):
        return 0.0
    else:
        mat = cosine_dist[select_indices, :][:, all_indices]
        repre = mat.max(axis=0)
        cover_score = repre.sum()
        return cover_score


def max_cover_query_step2(n_data, candidate_thresh, all_indices, cosine_dist):    
    # sample 
    select_indices = np.zeros((n_data, ), dtype=np.bool8)
    pre_cover_score = cover_score(cosine_dist, select_indices, all_indices)

    while (cover_score(cosine_dist, select_indices, all_indices) < candidate_thresh * all_indices.sum()):
        cover_score_list = np.zeros((n_data, ))
        for j in range(n_data):
            if not select_indices[j]:
                select_indices_temp = deepcopy(select_indices)
                select_indices_temp[j] = True
                cover_score_list[j] = cover_score(cosine_dist, select_indices_temp, all_indices)

        cover_score_list -= pre_cover_score
        next_sample_index = cover_score_list.argmax()
        select_indices[next_sample_index] = True
    
    return select_indices


def max_cover_query_step3(n_data, n_query, all_indices, cosine_dist):    
    # sample 
    select_indices = np.zeros((n_data, ), dtype=np.bool8)
    pre_cover_score = cover_score(cosine_dist, select_indices, all_indices)

    for _ in range(n_query):
        cover_score_list = np.zeros((n_data, ))
        for j in range(n_data):
            if not select_indices[j]:
                select_indices_temp = deepcopy(select_indices)
                select_indices_temp[j] = True
                cover_score_list[j] = cover_score(cosine_dist, select_indices_temp, all_indices)

        cover_score_list -= pre_cover_score
        next_sample_index = cover_score_list.argmax()
        select_indices[next_sample_index] = True
    
    return select_indices


def representative_annotation(n_data, n_query, feature, cosine_dist, candidate_thresh=0.9, n_clusters=3):
    if int(candidate_thresh * n_data) < n_query:
        raise ValueError(f"candidate_thresh is too small ({candidate_thresh})")

    # Step 1: Agglomerative Clustering
    agglo_cluster_label = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(feature)

    # Step 2: max-cover to form candidate set
    candidate_indices = np.zeros((n_data, ), dtype=np.bool8)
    for cluster_label in np.unique(agglo_cluster_label):
        cluster_indices = (agglo_cluster_label == cluster_label)

        cluster_sample_indices = max_cover_query_step2(
            n_data=n_data,
            candidate_thresh=candidate_thresh,
            all_indices=cluster_indices,
            cosine_dist=cosine_dist
        )

        candidate_indices = candidate_indices | cluster_sample_indices

    # print(candidate_indices.sum())

    # Step 3: max-cover to final sampling result
    ra_indices = max_cover_query_step3(
        n_data=n_data,
        n_query=n_query,
        all_indices=candidate_indices,
        cosine_dist=cosine_dist
    )

    return ra_indices


class RASelector(BaseSelector):
    def __init__(self, args, logger, batch_size, num_workers):
        super().__init__(args, logger, batch_size, num_workers)

    def calculate_scores(self, trainer, active_set):
        model = trainer.net
        model.eval()

        # concat ALL datasets
        label_dataset = active_set.label_dataset
        pool_dataset = active_set.pool_dataset
        core_list = label_dataset.im_idx
        pool_list = pool_dataset.im_idx
        all_list = pool_dataset.im_idx
        loader = get_al_loader(pool_dataset, self.batch_size, self.num_workers)

        # get encoder feature for every sample
        feature = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader, desc="ra feat")):
                # get data
                images = batch['images']
                images = images.to(trainer.device, dtype=torch.float32)

                # forward
                if trainer.data_parallel:
                    feats = model.module.get_enc_feature(images).cpu().numpy()      # [B, C]
                else:
                    feats = model.get_enc_feature(images).cpu().numpy()             # [B, C]
                feature.append(feats)

        feature = np.concatenate(feature, axis=0)                                   # [N, C]
        if self.args.distance == 'l2':
            feat_sim_mat = 1. / (pairwise_distances(feature, metric='euclidean') + 1e-8)
        elif self.args.distance == 'cosine':
            feat_sim_mat = cosine_similarity(feature)  # [N, N]
        else:
            raise NotImplementedError(f"{self.args.distance} is not supported for RA")

        return np.array(core_list), np.array(pool_list), np.array(all_list), feature, feat_sim_mat


    def select_next_batch(self, trainer, active_set, selection_count, selection_iter):
        # get list, feature and feature distance matrix
        core_list, pool_list, all_list, feature, feat_sim_mat = self.calculate_scores(trainer, active_set)

        # k-center greedy sampling
        newly_selected_idx = representative_annotation(
            n_data=len(pool_list),
            n_query=selection_count,
            feature=feature,
            cosine_dist=feat_sim_mat
        )

        # expand training set
        active_set.expand_training_set(pool_list[newly_selected_idx].tolist())

        # save feature and feature distance matrix
        save_path = osp.join(
            trainer.exp_dir, "record", f"coreset_round{selection_iter}.npz")
        np.savez_compressed(save_path, feat=feature, dist_mat=feat_sim_mat)