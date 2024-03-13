#This file is modified from official pycls repository to adapt in AL settings.
"""Meters."""

from collections import deque

import datetime
import numpy as np

from pycls.core.config import cfg
from pycls.utils.timer import Timer

import pycls.utils.logging as lu
import pycls.utils.metrics as metrics


def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{0:02},{1:02}:{2:02}:{3:02}'.format(days, hrs, mins, secs)


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, writer):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.writer = writer

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.loss.add_value(loss)
        self.lr = lr

        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
    
    def get_iter_stats(self, cur_epoch, cur_iter):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'train_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.epoch_iters),
            'top1_err': self.mb_top1_err.get_global_avg(),
            'loss': self.loss.get_global_avg(),
            'lr': self.lr,
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if ((cur_iter + 1) % cfg.LOG_PERIOD == 0) or ((cur_iter + 1) == self.epoch_iters):
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            # Log file
            lu.log_json_stats(stats)
            # Tensorboard
            for stat_key in ['top1_err', 'loss']:
                self.writer.add_scalar(f"train/{stat_key}_iter", stats[stat_key], cur_iter)

    def get_epoch_stats(self, cur_epoch):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch + 1) * self.epoch_iters
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            '_type': 'train_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'loss': avg_loss,
            'lr': self.lr,
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        # Log file
        lu.log_json_stats(stats)
        # Tensorboard
        for stat_key in ['top1_err', 'loss', 'lr']:
            self.writer.add_scalar(f"train/{stat_key}_epoch", stats[stat_key], cur_epoch)


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, max_iter, writer):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full test set)
        self.min_top1_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.writer = writer

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        iter_stats = {
            '_type': 'test_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.max_iter),
            'top1_err': self.mb_top1_err.get_global_avg(),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        # Log file
        lu.log_json_stats(stats)
        # Tensorboard
        self.writer.add_scalar(f"test/top1_err_iter", stats['top1_err'], cur_iter)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'test_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'min_top1_err': self.min_top1_err
        }
        return stats

    def log_epoch_stats(self, cur_epoch, test_metrics):
        stats = self.get_epoch_stats(cur_epoch)
        stats.update(test_metrics)
        # Log file
        lu.log_json_stats(stats)
        # Tensorboard
        self.writer.add_scalar(f"test/top1_acc", 100. - stats['top1_err'], cur_epoch)
        for k, v in test_metrics.items():
            self.writer.add_scalar(f"test/{k}", v, cur_epoch)


class ValMeter(object):
    """Measures Validation stats."""

    def __init__(self, max_iter, writer):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full Val set)
        self.min_top1_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_samples = 0
        self.writer = writer

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        iter_stats = {
            '_type': 'Val_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'iter': '{}/{}'.format(cur_iter + 1, self.max_iter),
            'top1_err': self.mb_top1_err.get_global_avg(),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        # Log file
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            '_type': 'Val_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            'top1_err': top1_err,
            'min_top1_err': self.min_top1_err
        }
        return stats

    def log_epoch_stats(self, cur_epoch, val_metrics):
        stats = self.get_epoch_stats(cur_epoch)
        stats.update(val_metrics)
        # Log file
        lu.log_json_stats(stats)
        # Tensorboard
        self.writer.add_scalar(f"val/top1_err", stats['top1_err'], cur_epoch)
        self.writer.add_scalar(f"val/min_top1_err", stats['min_top1_err'], cur_epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, cur_epoch)