import logging
import os
import os.path as osp
import random
import sys
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dateutil.relativedelta import relativedelta
from skimage.color import label2rgb
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if logger is None:
            print('\t'.join(entries))
        else:
            logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def active_percent_to_count(args, active_set):
    if args.count_per_round is None:
        percent_per_round = args.percent_per_round
        total_count = len(active_set.pool_dataset.im_idx) + len(active_set.label_dataset.im_idx)
        count_per_count = int(percent_per_round * total_count / 100)
    else:
        count_per_count = args.count_per_round
        total_count = len(active_set.pool_dataset.im_idx) + len(active_set.label_dataset.im_idx)
        args.percent_per_round = (count_per_count / total_count) * 100

    return count_per_count

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(exp_dir):
    # mkdir
    log_fname = osp.join(exp_dir, 'log.log')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT)
        ch = logging.StreamHandler(stream=sys.stdout)
        # ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        fh = logging.FileHandler(log_fname)
        # fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def timediff(t_start, t_end):
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def initialization(args):
    seed = args.seed

    # set random seed
    seed_everything(seed)

    # make exp dir 
    writer = SummaryWriter(args.exp_dir)
    os.makedirs(osp.join(args.exp_dir, "record"), exist_ok=True)

    # copy codes for every run
    def ignore(dir, files):
        return [f for f in files if f.startswith("__pycache__")]
    os.makedirs(osp.join(args.exp_dir, "codes"), exist_ok=True)
    for folder in ['active_selection', 'agent', 'dataloader', 'models', 'utils']:
        shutil.copytree(folder, osp.join(args.exp_dir, "codes", folder), ignore=ignore)
    for file in ['evaluate', 'train_sup', 'train_al']:
        shutil.copyfile(f"{file}.py", osp.join(args.exp_dir, "codes", f"{file}.py"))

    # init logger & save args
    logger = initialize_logging(args.exp_dir)
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    logger.info(' '.join(sys.argv))
    logger.info(args)

    return logger, writer


def finalization(t_start, val_result, test_result, logger, args):
    # End Experiment
    t_end = datetime.now()
    logger.info(f"{'%'*20} Experiment Report {'%'*20}")
    logger.info(f"0. AL Methods: {args.active_method}")
    logger.info(f"1. Takes: {timediff(t_start, t_end)}")
    logger.info(f"2. Log dir: {args.exp_dir} (with selection json & model checkpoint)")
    logger.info("3. Validation Metrics:")
    logger.info(pd.concat(val_result).to_string())
    logger.info("4. Test Metrics:")
    logger.info(pd.concat(test_result).to_string())
    logger.info(f"{'%'*20} Experiment End {'%'*20}")


def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
    torch.manual_seed(worker_id)


def colored_preds(label, num_classes):
    color_list = {
        0: [0, 0, 0],      # black
        1: [255, 0, 0],    # red
        2: [0, 255, 0],    # green
        3: [0, 0, 255],    # blue
        4: [255, 255, 0],  # yellow 
        5: [255, 0, 255],  # magenta
        6: [0, 255, 255],  # cyan
    }

    H, W = label.shape
    label_color = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        label_color[label == i] = color_list[i]
    
    return label_color


def visualize_sample_preds_slice(image, label, probs, preds, dice, asd, names, save_root):
    bsz = image.shape[0]
    image = image.cpu().numpy()
    label = label.cpu().numpy()
    probs = probs.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    C = probs.shape[1]

    if not osp.exists(save_root):
        os.mkdir(save_root)

    def s(img):
        img -= img.min()    # non-negative
        img /= img.max()    # scale 0-1
        return (img * 255).astype(np.uint8)

    for i in range(bsz):
        fig, axes = plt.subplots(2, 4)
        axes[0, 0].axis("off")
        axes[0, 0].set_title("image")
        axes[0, 0].imshow(s(image[i, 0]), cmap='gray')

        axes[0, 1].axis("off")
        axes[0, 1].set_title("label")
        axes[0, 1].imshow(label2rgb(label[i, 0]))

        axes[0, 2].axis("off")
        axes[0, 2].set_title("preds")
        axes[0, 2].imshow(label2rgb(preds[i, 0]))

        axes[0, 3].axis("off")
        axes[0, 3].set_title(f"prob 0")
        axes[0, 3].imshow(s(probs[i, 0]), cmap='gray')

        axes[1, 0].axis("off")
        axes[1, 0].set_title(f"prob 1 {dice[i, 0]:.3f} {asd[i, 0]:.3f}")
        axes[1, 0].imshow(s(probs[i, 1]), cmap='gray')

        axes[1, 1].axis("off")
        axes[1, 1].set_title(f"prob 2 {dice[i, 1]:.3f} {asd[i, 1]:.3f}")
        axes[1, 1].imshow(s(probs[i, 2]), cmap='gray')

        axes[1, 2].axis("off")
        axes[1, 2].set_title(f"prob 3 {dice[i, 2]:.3f} {asd[i, 2]:.3f}")
        axes[1, 2].imshow(s(probs[i, 3]), cmap='gray')
        
        axes[1, 3].axis("off")
        if C == 5:
            axes[1, 3].set_title(f"prob 4 {dice[i, 3]:.3f} {asd[i, 3]:.3f}")
            axes[1, 3].imshow(s(probs[i, 4]), cmap='gray')

        # save figure
        name = "_".join(names[i].split("/")[-2:])
        plt.tight_layout()
        plt.savefig(
            osp.join(save_root, name).replace(".npy", ".jpg"), dpi=200, bbox_inches='tight')
        plt.close()


def visualize_sample_preds_volume(image, label, probs, preds, dice, names, save_root):
    """
    Args:
        image: (D, 1, IMG_SIZE, IMG_SIZE) 
        label: (1, D, H, W) 
        probs: (D, C, H, W)
        preds: (1, D, H, W)
        dice: (C-1)
        names: len == 1
    """
    D, C, H, W = probs.shape

    image = F.interpolate(image, size=(H, W), mode="area").cpu().numpy()      # (D, 1, H, W)
    label = label.view(D, 1, H, W).cpu().numpy()
    preds = preds.view(D, 1, H, W).detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()

    os.makedirs(save_root, exist_ok=True)

    def s(img):
        img -= img.min()    # non-negative
        img /= img.max()    # scale 0-1
        return (img * 255).astype(np.uint8)

    fig, axes = plt.subplots(D, 3+C, figsize=((3+C)*2, D*2))
    for d in range(D):
        axes[d, 0].axis("off")
        axes[d, 0].set_title("image")
        axes[d, 0].imshow(s(image[d, 0]), cmap='gray')

        axes[d, 1].axis("off")
        axes[d, 1].set_title("label")
        axes[d, 1].imshow(label2rgb(label[d, 0]))

        axes[d, 2].axis("off")
        axes[d, 2].set_title("preds")
        axes[d, 2].imshow(label2rgb(preds[d, 0]))

        for c in range(C):
            title = f"prob {c}" if c == 0 else f"prob {c} {dice[c-1]:.3f}"
            axes[d, 3+c].axis("off")
            axes[d, 3+c].set_title(title)
            axes[d, 3+c].imshow(s(probs[d, c]), cmap='gray')

    # save figure
    name = "_".join(names[0].split("/")[-2:])
    plt.tight_layout()
    plt.savefig(
        osp.join(save_root, name).replace(".npy", ".jpg").replace(".npz", ".jpg"), 
        dpi=200, bbox_inches='tight')
    plt.close()


def check_ensemble_parameter(ens1, ens2):
    ens1_state_dict = ens1.state_dict()
    ens2_state_dict = ens2.state_dict()

    all_close = []
    for (k1, v1), (k2, v2) in zip(ens1_state_dict.items(), 
                                  ens2_state_dict.items()):
        assert k1 == k2
        all_close.append(torch.allclose(v1, v2))

    print(torch.tensor(all_close))