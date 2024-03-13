import torch
import torch.nn as nn


class CrossEntropyLossWeighted(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, n_classes=5):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = n_classes

    def one_hot(self, targets):
        targets_extend = targets.clone()
        targets_extend.unsqueeze_(1) # convert to Nx1xHxW
        one_hot = torch.cuda.FloatTensor(targets_extend.size(0), self.n_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)
        
        return one_hot
    
    def forward(self, inputs, targets):
        one_hot = self.one_hot(targets)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(one_hot, dim=(2, 3), keepdim=True)/torch.sum(one_hot)
        one_hot = weights*one_hot

        loss = self.ce(inputs, targets).unsqueeze(1) # shape is batch, 1, 256, 256
        loss = loss*one_hot

        return torch.sum(loss) / (torch.sum(weights)*targets.size(0)*targets.size(1))