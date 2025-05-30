# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import math
import time
from tqdm import tqdm

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from .ra_util import max_cover_query_step2, max_cover_query_step3

class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 


class RASampling():
    def __init__(self, cfg, dataObj):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TEST.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []
        
        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self, M1:torch.Tensor, M2:torch.Tensor):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        #print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        if self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'L2':
            M1_norm = (M1**2).sum(1).reshape(-1,1)
            M2_t = torch.transpose(M2, 0, 1)
            M2_norm = (M2**2).sum(1).reshape(1,-1)
            dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        elif self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'cosine':
            M1_norm = M1.norm(p=2, dim=1, keepdim=True)
            M2_norm = M2.norm(p=2, dim=1, keepdim=True)
            dot_product = torch.mm(M1, M2.t())
            cosine_similarity = dot_product / (torch.mm(M1_norm, M2_norm.t()) + 1e-8) # 加上一个小的常数防止除以0
            dists = 1 - cosine_similarity
        else:
            raise NotImplementedError(self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE)

        return dists
    
    def representative_annotation(self, n_data, n_query, feature, n_clusters=3, candidate_thresh=0.9):
        if int(candidate_thresh * n_data) < n_query:
            raise ValueError(f"candidate_thresh is too small ({candidate_thresh})")
        
        # Step 0: Calculate cosine similarity or 1 - L2 distance
        feature_tensor = torch.from_numpy(feature).cuda(self.cuda_id)
        dists = self.gpu_compute_dists(feature_tensor, feature_tensor)
        if self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'L2':
            sims = 1 / (1e-8 + dists)
        elif self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'cosine':
            sims = 1 - dists
        else:
            raise NotImplementedError(self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE)

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
                cosine_dist=sims.cpu().numpy()
            )

            candidate_indices = candidate_indices | cluster_sample_indices

        # print(candidate_indices.sum())

        # Step 3: max-cover to final sampling result
        ra_indices = max_cover_query_step3(
            n_data=n_data,
            n_query=n_query,
            all_indices=candidate_indices,
            cosine_dist=sims.cpu().numpy()
        )

        return np.nonzero(ra_indices)[0], np.nonzero(~ra_indices)[0]

    def query(self, lSet, uSet, clf_model, dataset):

        assert clf_model.training == False, "Classification model expected in training mode"
        assert clf_model.penultimate_active == True,"Classification model is expected in penultimate mode"    
        
        print("Extracting Uset Representations")
        ul_repr = self.get_representation(clf_model=clf_model, idx_set=uSet, dataset=dataset)
        print("ul_repr.shape: ",ul_repr.shape)
        
        # print("Solving K Center Greedy Approach")
        start = time.time()
        greedy_indexes, remainSet = self.representative_annotation(
            n_data=ul_repr.shape[0], 
            n_query=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, 
            feature=ul_repr
        )
        end = time.time()
        print("Time taken to solve Max Cover: {} seconds".format(end-start))
        activeSet = uSet[greedy_indexes]
        remainSet = uSet[remainSet]

        return activeSet, remainSet


class CoreSetMIPSampling():
    """
    Implements coreset MIP sampling operation
    """
    def __init__(self, cfg, dataObj, isMIP=False):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.isMIP = isMIP

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf_model = torch.nn.DataParallel(clf_model, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        
        #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TEST.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []
        
        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self, M1:torch.Tensor, M2:torch.Tensor):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        #print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        if self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'L2':
            M1_norm = (M1**2).sum(1).reshape(-1,1)
            M2_t = torch.transpose(M2, 0, 1)
            M2_norm = (M2**2).sum(1).reshape(1,-1)
            dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        elif self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE == 'cosine':
            M1_norm = M1.norm(p=2, dim=1, keepdim=True)
            M2_norm = M2.norm(p=2, dim=1, keepdim=True)
            dot_product = torch.mm(M1, M2.t())
            cosine_similarity = dot_product / (torch.mm(M1_norm, M2_norm.t()) + 1e-8) # 加上一个小的常数防止除以0
            dists = 1 - cosine_similarity
        else:
            raise NotImplementedError(self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE)

        return dists

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1).reshape((-1,1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        #order is important
        features = np.vstack((labeled,unlabeled))
        print("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end-start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i!=0 and i%500==0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices),dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:],axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)
        
        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))

        return greedy_indices-n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)]
        greedy_indices_counter = 0
        #move cpu to gpu
        labeled = torch.from_numpy(labeled).cuda(0)
        unlabeled = torch.from_numpy(unlabeled).cuda(0)

        print(f"[GPU] Labeled.shape: {labeled.shape}")
        print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
        print(f"[GPU] Distance Metirc: {self.cfg.ACTIVE_LEARNING.CORESET_DISTANCE}")
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()
        min_dist, _ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        print(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)
            
            min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices [greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        
        for i in tqdm(range(amount), desc = "Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            
            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))
            
            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices [greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices, remainSet, math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self, lSet, uSet, clf_model, dataset):

        assert clf_model.training == False, "Classification model expected in training mode"
        assert clf_model.penultimate_active == True,"Classification model is expected in penultimate mode"    
        
        print("Extracting Lset Representations")
        lb_repr = self.get_representation(clf_model=clf_model, idx_set=lSet, dataset=dataset)
        print("Extracting Uset Representations")
        ul_repr = self.get_representation(clf_model=clf_model, idx_set=uSet, dataset=dataset)
        
        print("lb_repr.shape: ",lb_repr.shape)
        print("ul_repr.shape: ",ul_repr.shape)
        
        if self.isMIP == True:
            raise NotImplementedError
        else:
            print("Solving K Center Greedy Approach")
            start = time.time()
            # greedy_indexes: list, remainSet: np.ndarray
            greedy_indexes, remainSet = self.greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            # greedy_indexes, remainSet = self.optimal_greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            end = time.time()
            print("Time taken to solve K center: {} seconds".format(end-start))
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]
        return activeSet, remainSet


class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = 0 if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble") else torch.cuda.current_device()
        self.dataObj = dataObj

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        # Used by bald acquisition
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        tempIdxSetLoader.dataset.no_aug = True
        preds = []
        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)

                temp_pred = clf_model(x)

                #  To get probabilities
                temp_pred = F.softmax(temp_pred,dim=1)
                preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tempIdxSetLoader.dataset.no_aug = False
        return preds


    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.
        
        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet   
        """

        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(uSet, np.ndarray), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(type(uSet))
        assert isinstance(budgetSize,int), "Expected budgetSize of type int whereas provided is dtype:{}".format(type(budgetSize))
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
            .format(len(uSet), budgetSize)

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]
        return activeSet, uSet


    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        clf_model.cuda(self.cuda_id)

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS), data=dataset)
        uSetLoader.dataset.no_aug = True
        n_uPts = len(uSet)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = self.get_predictions(clf_model=clf_model, idx_set=uSet, dataset=dataset)
            
            score_All += dropout_score

            # computing F_x
            dropout_score_log = np.log2(dropout_score + 1e-6) # Add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy

        U_X = G_X - F_X
        print("U_X.shape: ",U_X.shape)
        sorted_idx = np.argsort(U_X)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        clf_model.train()
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def dbal(self, budgetSize, uSet, clf_model, dataset):
        """
        Implements deep bayesian active learning where uncertainty is measured by 
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]
        
        In bayesian view, predictions are computed with the help of dropouts and 
        Monte Carlo approximation 
        """
        clf_model.cuda(self.cuda_id)

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        uSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS), data=dataset)
        uSetLoader.dataset.no_aug = True
        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        print("len usetLoader: {}".format(len(uSetLoader)))
        temp_i = 0
        
        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1
            x_u = x_u.type(torch.cuda.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS):
                with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    temp_op = clf_model(x_u)
                    # Till here z_op represents logits of p(y|x).
                    # So to get probabilities
                    temp_op = F.softmax(temp_op, dim=1)
                    z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS
            
            z_op = torch.from_numpy(z_op).cuda(self.cuda_id)
            entropy_z_op = entropy_loss(z_op, applySoftMax=False)
            
            # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
            u_scores.append(entropy_z_op.cpu().numpy())
            ptsProcessed += x_u.shape[0]
            
        u_scores = np.concatenate(u_scores, axis=0)
        sorted_idx = np.argsort(u_scores)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def uncertainty(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        Actually Least Confidence
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)
        
        clf = model.cuda()
        
        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = F.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
                temp_u_rank = 1 - temp_u_rank
                u_ranks.append(temp_u_rank.detach().cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def entropy(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = F.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1*torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def margin(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)   # [128, 3, 224, 224]

                temp_u_rank = F.softmax(clf(x_u), dim=1)    # [128, 10]
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)   # [128, 10]
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]  # [128]
                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value  
                difference = -1 * difference 
                u_ranks.append(difference.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)   # [len(uSet)]
        # Now u_ranks has shape: [U_Size, ]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        sorted_idx = np.argsort(u_ranks)[::-1]  # [len(uSet), ], uncertainty value to index
        activeSet = sorted_idx[:budgetSize]  # [budget_size, ]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def badge(self, budgetSize, lSet, uSet, model, dataset):

        """
        Deep batch active learning by diverse, uncertain gradient lower bounds.
        """
        import time
        from monai.networks import one_hot
        from sklearn.cluster import kmeans_plusplus
        
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        grads = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                scores, outputs = clf(x_u)   # [B, D], [B, C]
                probs = F.softmax(outputs, dim=1) # [B, C]
                pseudo_label = torch.argmax(probs, dim=1)   # [B, ]
                pseudo_label_one_hot = one_hot(
                    pseudo_label.unsqueeze(1), num_classes=num_classes, dim=1)  # [B, C]
                ce_grad = (probs - pseudo_label_one_hot).unsqueeze(2) * scores.unsqueeze(1) # [B, C, D]
                ce_grad = ce_grad.flatten(1)    # [B, C*D]
                grads.append(ce_grad.detach().cpu().numpy())
        grads = np.concatenate(grads, axis=0) # [U_Size, C*D]

        # index of grads serve as key to refer in u_idx
        print(f"grads.shape: {grads.shape}")
        # we add -1 for reversing the sorted array
        t = time.time()
        print(f"K-Means++ start... ")
        # _, activeSet = kmeans_plusplus(X=grads, n_clusters=10)    # for debug
        _, activeSet = kmeans_plusplus(X=grads, n_clusters=budgetSize)
        print(f"K-Means++ end, spend {(time.time() - t):.2f}s ")
        remainSet = np.setdiff1d(np.arange(grads.shape[0]), activeSet)

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

