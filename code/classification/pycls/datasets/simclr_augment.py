# Modified from the source: https://github.com/sthalles/PyTorch-BYOL
# Previous owner of this file: Thalles Silva
 
from torchvision import transforms
from .utils.gaussian_blur import GaussianBlur

def get_simclr_ops(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    ops = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * input_shape)),
    ]
    return ops


