from monai.networks import one_hot

def dice(output, target, num_classes=None, eps:float=1e-5):
    """
    output: [B, 1, H, W] or [B, C, H, W]
    target: [B, 1, H, W] or [B, C, H, W]
    dsc: [B, C]
    """
    # one hot encoding
    if output.shape[1] == 1:
        output = one_hot(output, num_classes=num_classes, dim=1)
    if target.shape[1] == 1:
        target = one_hot(target, num_classes=num_classes, dim=1)
    
    # compute dice
    target = target.float()
    num = 2 * (output * target).sum(dim=(2, 3))
    den = (output + target).sum(dim=(2, 3)) + eps
    dsc = num / den

    return dsc

