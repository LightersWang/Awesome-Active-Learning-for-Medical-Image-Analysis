# Modified from the source: https://github.com/JiaxinZhuang/Skin-Lesion-Recognition.Pytorch/blob/master/src/trainer.py
 
from torch import Tensor
from torchvision import transforms

class ZeroOneNormalize:
    def __call__(self, img:Tensor):
        return img.float().div(255)

def get_isic_ops(image_size, is_train, mode='isic'):

    if is_train:
        if mode == 'isic':
            ops = transforms.Compose([
                    # ↓ if the image reading is too slow, you can comment this line and resize the images to 300x300 offline 
                    transforms.Resize([300, 300], antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                    transforms.RandomRotation([-180, 180]),
                    transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
                    transforms.RandomCrop([image_size, image_size]),
                    ZeroOneNormalize(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
        elif mode == 'hflip':
            ops = transforms.Compose([
                    transforms.Resize([image_size, image_size], antialias=True),
                    transforms.RandomHorizontalFlip(),
                    ZeroOneNormalize(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError
    else:
        ops = transforms.Compose([
            transforms.Resize([image_size, image_size], antialias=True),
            ZeroOneNormalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        ])

    return ops


