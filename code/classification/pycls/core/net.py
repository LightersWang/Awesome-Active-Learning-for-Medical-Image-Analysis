#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for manipulating networks."""

import itertools

import numpy as np
import torch
from pycls.core.config import cfg


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def complexity(model):
    """Compute model complexity (model can be model instance or model class)."""
    size = cfg.TRAIN.IM_SIZE
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    cx = unwrap_model(model).complexity(cx)
    return {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}


def smooth_one_hot_labels(labels):
    """Convert each label to a one-hot vector."""
    n_classes, label_smooth = cfg.MODEL.NUM_CLASSES, cfg.TRAIN.LABEL_SMOOTHING
    err_str = "Invalid input to one_hot_vector()"
    assert labels.ndim == 1 and labels.max() < n_classes, err_str
    shape = (labels.shape[0], n_classes)
    neg_val = label_smooth / n_classes
    pos_val = 1.0 - label_smooth + neg_val
    labels_one_hot = torch.full(shape, neg_val, dtype=torch.float, device=labels.device)
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)
    return labels_one_hot


class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]


def mixup(inputs, labels):
    """Apply mixup to minibatch (https://arxiv.org/abs/1710.09412)."""
    alpha = cfg.TRAIN.MIXUP_ALPHA
    assert labels.shape[1] == cfg.MODEL.NUM_CLASSES, "mixup labels must be one-hot"
    if alpha > 0:
        m = np.random.beta(alpha, alpha)
        permutation = torch.randperm(labels.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation, :]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    return inputs, labels, labels.argmax(1)
