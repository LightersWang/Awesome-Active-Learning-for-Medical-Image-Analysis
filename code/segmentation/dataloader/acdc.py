import os.path as osp
import h5py
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.utils.data as data
import monai.transforms as transforms


tensor_size_in_gb = lambda tensor: tensor.element_size() * tensor.numel() / (1024 ** 3)

class ACDC(data.Dataset):
    """ACDC dataset"""
    def __init__(self, args, root, datalist, split='train', transform=None):
        assert split in ['train', 'test', 'val', 'active-label', 'active-pool']

        self.args = args
        self.split = split
        self.root = osp.expanduser(root)
        self.spacing_df = pd.read_csv(osp.join(root, 'acdc_spacing.csv'), index_col=0)

        # Use default transform
        if split in ["train", "active-label"]:
            self.transform = self.get_train_transform_rand()
            self.get_spacing = lambda id: self.spacing_df.loc[id][:2].values
            self.preload_pattern = 'data/slices/*.h5'
            self.preload_transform = self.get_train_transform_deterministic()
        elif split in ["active-pool"]:
            self.transform = None   # pool dataset shouldn't perform any random transforms
            self.get_spacing = lambda id: self.spacing_df.loc[id][:2].values
            self.preload_pattern = 'data/slices/*.h5'
            self.preload_transform = self.get_train_transform_deterministic()
        elif split in ["val", "test"]:
            self.transform = None
            self.get_spacing = lambda id: self.spacing_df.loc[id].values
            self.preload_pattern = 'data/*.h5'
            self.preload_transform = self.get_val_transform()
        else:
            raise NotImplementedError(split)

        # im_idx contains the list of each image paths
        self.im_idx = []
        if datalist is not None:
            valid_list = np.loadtxt(datalist, dtype=str).tolist()
            for data_fname in valid_list:
                img_fullname = osp.join(self.root, data_fname)   # img & lbl 
                self.im_idx.append([img_fullname, ])
        
        # preload all samples (image & label) for faster training speed
        self.images = []
        self.labels = []
        self.fnames = []
        total_gb = 0.
        datalist_name = osp.basename(datalist) if datalist is not None else ''
        pbar = tqdm(glob(osp.join(root, self.preload_pattern)), 
                    desc=f"ACDC {split} {datalist_name} {total_gb:.4f} GB")
        for data_fname in pbar:
            data = h5py.File(data_fname, 'r')
            image = data['image'][:]
            label = data['label'][:]
            sample = self.preload_transform({'images': image, 'labels': label})
            transformed_image = sample['images'].to(torch.float32)
            transformed_label = sample['labels'].to(torch.long)

            # save to RAM
            self.images.append(transformed_image)
            self.labels.append(transformed_label)
            self.fnames.append(data_fname)

            # logging storage space
            total_gb += tensor_size_in_gb(sample['images'])
            total_gb += tensor_size_in_gb(sample['labels'])
            pbar.set_description(f"ACDC {split} {datalist_name} {total_gb:.4f} GB")

        self.fnames = np.array(self.fnames)
        if split in ["train", "active-label", "active-pool"]:
            self.images = torch.stack(self.images, dim=0)
            self.labels = torch.stack(self.labels, dim=0)


    def __getitem__(self, index):
        """
            Train mode: image and label shape is (1, H, W)
            Val mode: image and label shape is (1, D, H, W)
        """
        data_fname = self.im_idx[index][0]
        data_index = np.where(self.fnames == data_fname)[0][0]
        image = self.images[data_index]
        label = self.labels[data_index]
        sample = {'images': image, 'labels': label}

        if self.transform is not None:
            sample = self.transform(sample)

        # spacing
        patient_id = data_fname.split('/')[-1].split('_')[0]
        spacing = self.get_spacing(patient_id)
        sample.update({
            'fnames': self.im_idx[index], 
            'indices': torch.tensor(index),
            'spacing': torch.tensor(spacing)
        })

        return sample


    def __len__(self):
        return len(self.im_idx)


    # resize 2D image and label to tensors of shape (256, 256) 
    def get_train_transform_deterministic(self):
        H = W = self.args.img_size
        train_transform = transforms.Compose([
            transforms.EnsureChannelFirstd(keys=['images', 'labels'], channel_dim='no_channel'),
            transforms.Resized(keys=['images'], spatial_size=(H, W)),
            transforms.Resized(keys=['labels'], spatial_size=(H, W), mode='nearest'),
            transforms.EnsureTyped(keys=["images", 'labels']),
        ])
        return train_transform

    # random data augmentation
    def get_train_transform_rand(self):
        # H = W = self.args.img_size
        train_transform = transforms.Compose([
            transforms.RandFlipd(keys=['images', 'labels'], prob=0.5),
            transforms.RandRotate90d(keys=['images', 'labels'], prob=0.5),
            transforms.RandRotated(keys=['images', 'labels'], range_x=0.4, prob=0.5),
        ])
        return train_transform

    # return 3D image and label, resize image to (-1, 256, 256)
    def get_val_transform(self):
        H = W = self.args.img_size
        val_transform = transforms.Compose([
            transforms.EnsureChannelFirstd(keys=['images', 'labels'], channel_dim='no_channel'),
            transforms.Resized(keys=['images'], spatial_size=(-1, H, W)),
            transforms.EnsureTyped(keys=["images", 'labels']),
        ])
        return val_transform
