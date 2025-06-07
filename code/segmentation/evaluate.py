import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import os
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from medpy.metric import dc
from monai.networks import one_hot
from torch.utils.data import DataLoader
from medpy.metric import dc, asd

from dataloader import get_dataset
from models import get_model
from utils.common import AverageMeter


def infer_volume(args, net, images, labels, dice_meters, asd_meters, spacing=None):
    """
        images: [1, 1, D, IMG_SIZE, IMG_SIZE]
        labels: [1, 1, D, H, W]
        IMG_SIZE usually equals to 256
    """
    assert args.batch_size == 1, "batch size of volume infer must be 1"

    # turn 3D input to batched 2D input
    IMG_SIZE, C = args.img_size, args.num_classes
    D, H, W = labels.shape[2:]
    images = images.view(D, 1, IMG_SIZE, IMG_SIZE)              # (D, 1, IMG_SIZE, IMG_SIZE) 
    labels = labels.view(1, D, H, W)                            # (1, D, H, W)

    # forward, resize probability to original size
    outputs = net(images)                                       # (D, C, IMG_SIZE, IMG_SIZE)
    probs = torch.softmax(outputs, dim=1)                       # (D, C, IMG_SIZE, IMG_SIZE)
    probs = F.interpolate(probs, size=(H, W), mode="area")      # (D, C, H, W)
    preds = torch.argmax(probs, dim=1, keepdim=True)            # (D, 1, H, W)
    preds = preds.view(1, D, H, W)                              # (1, D, H, W)

    # to one-hot
    preds_onehot = one_hot(preds, num_classes=C, dim=0).cpu().numpy()
    labels_onehot = one_hot(labels, num_classes=C, dim=0).cpu().numpy()

    # update dice meter & skip empty class
    for c in range(1, C):
        if labels_onehot[c-1].any():
            dice_c = dc(preds_onehot[c], labels_onehot[c])
            dice_meters[c-1].update(dice_c)
    
    # update asd meter: both label and pred should have binary object
    for c in range(1, C):
        if preds_onehot[c].any() and labels_onehot[c].any():
            voxelspacing = np.roll(spacing[0], shift=1)     # spacing of [x, y, z] -> [z, x, y]
            asd_bc = asd(preds_onehot[c], labels_onehot[c], voxelspacing=voxelspacing)
            # asd_bc = asd(preds_onehot[c], labels_onehot[c])
            asd_meters[c-1].update(asd_bc)


def evaluation(args, net, dataloader, device):
    net.eval()

    # mean_dice_meter = AverageMeter('mean_Dice', ':8.4f')
    dice_meters = [AverageMeter(f'Dice_{i+1}', ':8.4f') for i in range(args.num_classes - 1)]
    asd_meters = [AverageMeter(f'ASD_{i+1}', ':8.4f') for i in range(args.num_classes - 1)]

    with torch.no_grad():
        for batch in dataloader:
            # get data
            images = batch['images']    # (B, 1, 256, 256) or (B, 1, D, 256, 256) 
            labels = batch['labels']    # (B, 1, H, W) or (B, 1, D, H, W) 
            spacing = batch['spacing'].cpu().numpy()  if 'spacing' in batch else None
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            infer_volume(args, net, images, labels, dice_meters, asd_meters, spacing)

        mean_dice = np.mean([meter.avg for meter in dice_meters]) * 100
        mean_asd = np.mean([meter.avg for meter in asd_meters]) 

        return mean_dice, mean_asd


def find_all_ckpts(ckpt_dir):
    """Recursively find all .tar checkpoint files"""
    ckpt_files = []
    for root, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith('.tar'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


def parse_seed_episode(ckpt_path):
    """
    Parse seed and episode from path.
    Example: .../seed1000/.../checkpoint01.tar -> (1000, 1)
    """
    seed_match = re.search(r'seed(\d+)', ckpt_path)
    episode_match = re.search(r'checkpoint(\d+)', ckpt_path)
    seed = int(seed_match.group(1)) if seed_match else None
    episode = int(episode_match.group(1)) if episode_match else None
    return seed, episode


def build_ckpt_table(ckpt_files):
    """Build a DataFrame of checkpoint paths, rows are seeds, columns are episodes"""
    records = []
    for path in ckpt_files:
        seed, episode = parse_seed_episode(path)
        if seed is not None and episode is not None:
            records.append({'seed': seed, 'episode': episode, 'path': path})
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid checkpoint files found")
    seeds = sorted(df['seed'].unique())
    episodes = sorted(df['episode'].unique())
    table = pd.DataFrame(index=seeds, columns=episodes)
    for _, row in df.iterrows():
        table.at[row['seed'], row['episode']] = row['path']
    return table


def evaluate_ckpt_table(args, ckpt_table):
    """Infer all checkpoints, return mean_dice and mean_asd DataFrames"""
    mean_dice_table = pd.DataFrame(index=ckpt_table.index, columns=ckpt_table.columns)
    mean_asd_table = pd.DataFrame(index=ckpt_table.index, columns=ckpt_table.columns)
    for seed in ckpt_table.index:
        for episode in ckpt_table.columns:
            ckpt_path = ckpt_table.at[seed, episode]
            if ckpt_path is None:
                continue
            args.ckpt_dir = ckpt_path
            # reset random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            # inference
            mean_dice, mean_asd = main(args)
            mean_dice_table.at[seed, episode] = mean_dice
            mean_asd_table.at[seed, episode] = mean_asd
    return mean_dice_table, mean_asd_table


def add_stat_rows(table):
    """Add Mean, Std, and Result rows"""
    mean_row = table.astype(float).mean(axis=0)
    std_row = table.astype(float).std(axis=0)
    table.loc['Mean'] = mean_row
    table.loc['Std'] = std_row
    result_row = [f"{mean:.2f}±{std:.2f}" for mean, std in zip(mean_row, std_row)]
    table.loc['Result'] = result_row
    return table


def main(args):
    # If ckpt_dir is a directory, do batch inference
    ckpt_files = find_all_ckpts(args.ckpt_dir)
    print(f"Found {len(ckpt_files)} checkpoint files")
    ckpt_table = build_ckpt_table(ckpt_files)

    # Load dataset and dataloader only once
    device = torch.device(args.device)
    test_set = get_dataset(args, args.split)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize result tables
    mean_dice_table = pd.DataFrame(index=ckpt_table.index, columns=ckpt_table.columns)
    mean_asd_table = pd.DataFrame(index=ckpt_table.index, columns=ckpt_table.columns)
    
    # Inference and fill tables
    for seed in ckpt_table.index:
        for episode in tqdm(ckpt_table.columns, desc=f'Seed {seed}:'):
            ckpt_path = ckpt_table.at[seed, episode]
            if ckpt_path is None:
                continue
            print(f"\n[Seed {seed}][Episode {episode}] Inference: {ckpt_path}...")
            args.ckpt_dir = ckpt_path
            torch.manual_seed(seed)
            np.random.seed(seed)
            # Load weights and model for each checkpoint
            checkpoint = torch.load(args.ckpt_dir, map_location=device)
            net = get_model(args).to(device)
            net_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                new_k = k.replace(".trans_convs", ".upsamples").replace("module.", "")
                net_state_dict.update({new_k: v})
            net.load_state_dict(net_state_dict)
            mean_dice, mean_asd = evaluation(args, net, test_loader, device)
            mean_dice_table.at[seed, episode] = mean_dice
            mean_asd_table.at[seed, episode] = mean_asd

    # Statistics and print
    print("\n=== Mean Dice ===")
    mean_dice_table = add_stat_rows(mean_dice_table)
    print(mean_dice_table)
    
    print("\n=== Mean ASD ===")
    mean_asd_table = add_stat_rows(mean_asd_table)
    print(mean_asd_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument("--ckpt_dir", type=str, help="folder dir includes checkpoints or path to ckpt")

    # model 
    parser.add_argument("--model", type=str, default="unet_plain", help="model name for loading pretrained weight")
    parser.add_argument("--unet_channels", type=int, nargs='+', default=[32, 64, 128, 256, 512],
                        help="#channels of every level in u-net, last is bottleneck dim")
    parser.add_argument('--input_dimension', type=int, default=2, choices=[2, 3], 
                        help="spatial dimension of input & network")
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument("--normalization", type=str, default='batch', choices=['instance', 'batch'],
                        help="type of normalization in unet conv block")
    parser.add_argument("--dropout_prob", type=float, default=0.1, 
                        help="dropout probability for all conv blocks in unet")
    parser.add_argument('--deep_supervision', action='store_true', default=False, 
                        help="whether use deep supervision in unet")
    parser.add_argument('--deep_supervision_layer', type=int, default=3,
                        help='last x layers for deep supervision')

    # dataset
    parser.add_argument("--dataset", type=str, default="ACDC", help="dataset name for evaluation")
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes of the specified dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--batch_size", type=int, default=1)

    # misc
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cuda:1"])
    parser.add_argument("--max_num_slice", type=int, default=128, 
                        help="max number of infered slices, higher number may cause GPU out-of-memory")

    args = parser.parse_args()
    main(args)
