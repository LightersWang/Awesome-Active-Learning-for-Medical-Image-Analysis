import os
import os.path as osp
import json


class ActiveDataset:
    def __init__(self, args, pool_dataset, label_dataset):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.pool_dataset = pool_dataset
        self.label_dataset = label_dataset


    def expand_training_set(self, paths):
        self.label_dataset.im_idx.extend(paths)
        for x in paths:
            self.pool_dataset.im_idx.remove(x)


    def get_fraction_of_labeled_data(self):
        label_num = len(self.label_dataset.im_idx)
        pool_num = len(self.pool_dataset.im_idx)
        return label_num / float(label_num + pool_num)


    def dump_datalist(self):
        os.makedirs(osp.join(self.args.exp_dir, "datalist"), exist_ok=True)
        datalist_path = osp.join(
            self.args.exp_dir, "datalist", f'datalist_{self.selection_iter:02d}.json')
        store_data = {
            'label_im_idx': self.label_dataset.im_idx,
            'pool_im_idx': self.pool_dataset.im_idx,
        }
        with open(datalist_path, "w") as f:
            json.dump(store_data, f)


    def load_datalist(self):
        datalist_path = osp.join(
            self.args.exp_dir, "datalist", f'datalist_{self.selection_iter:02d}.json')
        with open(datalist_path, "rb") as f:
            json_data = json.load(f)
        self.label_dataset.im_idx = json_data['label_im_idx']
        self.pool_dataset.im_idx = json_data['pool_im_idx']


    def dump_initial_datalist(self, init_datalist):
        os.makedirs(osp.dirname(init_datalist), exist_ok=True)
        store_data = {
            'label_im_idx': self.label_dataset.im_idx,
            'pool_im_idx': self.pool_dataset.im_idx,
        }
        with open(init_datalist, "w") as f:
            json.dump(store_data, f)


    def load_initial_datalist(self, init_datalist):
        with open(init_datalist, "rb") as f:
            json_data = json.load(f)
        self.label_dataset.im_idx = json_data['label_im_idx']
        self.pool_dataset.im_idx = json_data['pool_im_idx']


    def get_trainset(self):
        return self.label_dataset
