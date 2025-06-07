import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

import torch.nn.functional as F
from monai.utils import set_determinism

# local
for path in ['..', '.']:
    if path not in sys.path:
        sys.path.insert(0, path)

import pycls.core.builders as model_builder
from pycls.core.config import cfg
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.metrics as mu
from pycls.utils.meters import TestMeter

def argparser():
    parser = argparse.ArgumentParser(description='Model Inference - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--ckpt', dest='ckpt_dir', help='Checkpoint dir', required=True, type=str)
    return parser

def find_ckpt_files(ckpt_dir):
    ckpt_files = []
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith('.pyth'):
                ckpt_files.append(os.path.join(root, f))
    return ckpt_files

def parse_seed_episode(path):
    # seed(\d+)
    seed_match = re.search(r'seed(\d+)', path)
    episode_match = re.search(r'episode_(\d+)', path)
    seed = int(seed_match.group(1)) if seed_match else None
    episode = int(episode_match.group(1)) if episode_match else None
    return seed, episode

def build_ckpt_table(ckpt_files):
    # Parse all seeds and episodes
    records = []
    seeds = set()
    episodes = set()
    for f in ckpt_files:
        seed, episode = parse_seed_episode(f)
        if seed is not None and episode is not None:
            records.append((seed, episode, f))
            seeds.add(seed)
            episodes.add(episode)
    seeds = sorted(seeds)
    episodes = sorted(episodes)
    # Build DataFrame
    table = pd.DataFrame(index=seeds, columns=episodes)
    for seed, episode, f in records:
        table.at[seed, episode] = f
    return table

def build_metric_table(seeds, episodes):
    return pd.DataFrame(index=seeds, columns=episodes)

def main(cfg, ckpt_dir):
    # Setting up GPU args
    loader_cfg = {"num_workers": 0, "pin_memory": True}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)
        print(f'Assigned random RNG_SEED: {cfg.RNG_SEED}')
    set_determinism(cfg.RNG_SEED)

    # 1. Find all checkpoint files
    ckpt_files = find_ckpt_files(ckpt_dir)
    print(f"Found {len(ckpt_files)} checkpoint files.")

    # 2. Parse seed and episode
    ckpt_table = build_ckpt_table(ckpt_files)

    # 3. Build metric table
    metric_table = build_metric_table(ckpt_table.index, ckpt_table.columns)

    # 4. Inference and fill the table
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('.'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    test_data, _ = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED, **loader_cfg)

    for seed in ckpt_table.index:
        for episode in tqdm(ckpt_table.columns, desc=f'Seed {seed}:'):
            ckpt_path = ckpt_table.at[seed, episode]
            if ckpt_path is not None:
                print(f"\n[Seed {seed}][Episode {episode}] Inference: {ckpt_path}...")
                test_metrics = test_model(test_loader, ckpt_path, cfg, cur_episode=0)
                main_metric = cfg.DATASET.METRICS[0]
                metric_table.at[seed, episode] = test_metrics.get(main_metric, None)

        print(metric_table)

    # 5. Calculate mean and std
    mean_row = metric_table.astype(float).mean(axis=0)
    std_row = metric_table.astype(float).std(axis=0)
    metric_table.loc['Mean'] = mean_row
    metric_table.loc['Std'] = std_row

    # 6. Build Result row
    result_row = []
    for col in metric_table.columns:
        mean = mean_row[col]
        std = std_row[col]
        if pd.isna(mean) or pd.isna(std):
            result_row.append('')
        else:
            result_row.append(f"{mean:.2f}±{std:.2f}")
    metric_table.loc['Result'] = result_row

    print("\nFinal Metric Table:")
    print(metric_table)


def test_model(test_loader, checkpoint_file, cfg, cur_episode):
    test_meter = TestMeter(len(test_loader), writer=None)
    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    test_metrics = test_epoch(test_loader, model, test_meter, cur_episode)
    return test_metrics


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    # for metrics
    test_metrics = {}
    misclassifications, totalSamples = 0., 0.
    test_probs, test_labels = [], []

    with torch.no_grad():
        for cur_iter, (inputs, labels) in enumerate(test_loader):
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)

            if ('AUC' in cfg.DATASET.METRICS) or ('F1' in cfg.DATASET.METRICS):
                # Compute the probabilities after softmax
                probs = F.softmax(preds, dim=1)
                test_probs.append(probs)
                test_labels.append(labels)

            # Compute the errors
            top1_err = mu.topk_errors(preds, labels, [1,])[0].item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0) * cfg.NUM_GPUS

            # Update and log stats
            test_meter.iter_toc()
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.iter_tic()

    # metric calculation
    test_probs = torch.cat(test_probs, dim=0).cpu().numpy() if test_probs else None
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy() if test_labels else None
    if 'Accuracy' in cfg.DATASET.METRICS:
        test_acc = 100. - (misclassifications / totalSamples)
        test_metrics.update({'Accuracy': test_acc})
    if 'AUC' in cfg.DATASET.METRICS and test_probs is not None:
        test_auc = roc_auc_score(test_labels, test_probs[:, 1])
        test_auc = max(test_auc, 1. - test_auc) * 100
        test_metrics.update({'AUC': test_auc})
    if 'F1' in cfg.DATASET.METRICS and test_probs is not None:
        # find optimal threshold by maximizing f1 score
        precision, recall, thresh = precision_recall_curve(test_labels, test_probs[:, 1])
        fscore = (2 * precision * recall) / (precision + recall)
        thresh_opt = thresh[np.argmax(fscore)].astype(np.float64)
        test_pred = np.array(test_probs[:, 1] > thresh_opt, dtype=int)

        test_f1 = f1_score(test_labels, test_pred)
        test_metrics.update({'F1': test_f1})

    # Log epoch stats
    test_meter.reset()

    return test_metrics


if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    main(cfg, args.ckpt_dir)
    print()