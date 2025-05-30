# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import numpy as np 
import pycls.utils.logging as lu
from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler, RASampling

logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj, cfg=cfg)
        self.cfg = cfg
        
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
            activeSet, uSet = self.sampler.random(
                uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, lSet=lSet, uSet=uSet, 
                model=clf_model, dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, lSet=lSet, uSet=uSet, 
                model=clf_model, dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, lSet=lSet, uSet=uSet, 
                model=clf_model, dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["coreset", "coreset_L2", "coreset_cosine"]:
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(
                lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["ra", "ra_L2", "ra_cosine"]:
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            raSampler = RASampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = raSampler.query(
                lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["dbal", "DBAL"]:
            activeSet, uSet = self.sampler.dbal(
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, 
                clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["bald", "BALD"]:
            activeSet, uSet = self.sampler.bald(
                budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, 
                clf_model=clf_model, dataset=trainDataset)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "badge":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            activeSet, uSet = self.sampler.badge(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, lSet=lSet, uSet=uSet, 
                                                  model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
        
