import numpy as np

from tqdm import tqdm
from copy import deepcopy


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


# def max_cover_query_step3(n_data, n_query, all_indices, cosine_dist):    
#     # sample 
#     select_indices = np.zeros((n_data, ), dtype=np.bool8)
#     pre_cover_score = cover_score(cosine_dist, select_indices, all_indices)

#     for _ in tqdm(range(n_query), desc='max_cover_query_step3'):
#         cover_score_list = np.zeros((n_data, ))
#         for j in range(n_data):
#             if not select_indices[j]:
#                 select_indices[j] = True
#                 cover_score_list[j] = cover_score(cosine_dist, select_indices, all_indices)
#                 select_indices[j] = False

#         cover_score_list -= pre_cover_score
#         next_sample_index = cover_score_list.argmax()
#         select_indices[next_sample_index] = True
    
#     return select_indices


def max_cover_query_step3(n_data, n_query, all_indices, cosine_dist):
    cosine_all = cosine_dist[:, all_indices]
    current_max = np.zeros(cosine_all.shape[1], dtype=np.float32)
    current_sum = 0.0
    select_indices = np.zeros(n_data, dtype=np.bool8)
    # select_indices[all_indices] = True
    
    for _ in tqdm(range(n_query), desc='Optimized max_cover_query_step3'):
        mask = ~select_indices
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            break  
        
        rows = cosine_all[candidates]
        gains = np.sum(np.maximum(current_max, rows), axis=1) - current_sum
        # gains = np.sum(rows, axis=1)
        
        best_idx = np.argmax(gains)
        best_j = candidates[best_idx]
        
        select_indices[best_j] = True
        current_max = np.maximum(current_max, cosine_all[best_j])
        current_sum += gains[best_idx]
    
    return select_indices