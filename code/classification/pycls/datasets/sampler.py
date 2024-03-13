# This file is directly taken from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from torch.utils.data.sampler import Sampler


class IndexedSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_idxes (Dataset indexes): dataset indexes to sample from
    """

    def __init__(self, data_idxes, isDebug=False):
        if isDebug: print("========= my custom squential sampler =========")
        self.data_idxes = data_idxes

    def __iter__(self):
        return (self.data_idxes[i] for i in range(len(self.data_idxes)))

    def __len__(self):
        return len(self.data_idxes)
