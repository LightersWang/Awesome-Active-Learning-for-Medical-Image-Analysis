import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class NCT_CRC_HE_100K(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, test_transform, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.no_aug:
            if self.test_transform is not None:
                sample = self.test_transform(sample)          
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


tensor_size_in_gb = lambda tensor: tensor.element_size() * tensor.numel() / (1024 ** 3)

class ISIC2020(Dataset):
    def __init__(self, root, train, transform, test_transform):
        super(ISIC2020, self).__init__()
        self.root = root
        self.is_train = train
        if self.is_train:
            self.split_df = pd.read_csv(os.path.join(root, 'trainval_split.csv'))
        else:
            self.split_df = pd.read_csv(os.path.join(root, 'test_split.csv'))

        self.transform = transform
        self.test_transform = test_transform
        self.no_aug = False

        self.images = []
        self.targets = []
        self.image_names = []
        total_gb = 0.
        pbar = tqdm(list(self.split_df.index), desc=f"ISIC 2020 {total_gb:.4f} GB")
        for index in pbar:
            # read image and target
            img_name = self.split_df.iloc[index]['image_name']
            target = self.split_df.iloc[index]['target']
            img_path = os.path.join(
                self.root, 'ISIC_2020_Training_JPEG_300x300', 'train', f'{img_name}.jpg')
            img = read_image(img_path, mode=ImageReadMode.RGB)

            # save to RAM
            self.images.append(img)
            self.targets.append(target)
            self.image_names.append(img_name)

            # logging storage space
            total_gb += tensor_size_in_gb(img)
            pbar.set_description(f"ISIC 2020 {total_gb:.4f} GB")

        # to tensor or ndarray
        self.images = torch.stack(self.images, dim=0)
        self.targets = torch.tensor(self.targets)
        self.image_names = np.array(self.image_names)
    

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.images[index]
        target = self.targets[index]

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target
    
    def __len__(self):
        return self.split_df.shape[0]