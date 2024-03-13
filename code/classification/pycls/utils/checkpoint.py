#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os
from shutil import copyfile

import torch
from pycls.core.config import cfg
from pycls.core.net import unwrap_model


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"

# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir(episode_dir):
    """Retrieves the location for storing checkpoints."""
    return os.path.join(episode_dir, _DIR_NAME)


def get_checkpoint(epoch, episode_dir):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    # return os.path.join(get_checkpoint_dir(), name)
    return os.path.join(episode_dir, name)


def get_checkpoint_best(episode_dir):
    """Retrieves the path to the best checkpoint file."""
    return os.path.join(episode_dir, "model.pyth")


def get_last_checkpoint(episode_dir):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir(episode_dir)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint(episode_dir):
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir(episode_dir)
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def save_checkpoint(info, model_state, optimizer_state, epoch, cfg):
    
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    # if not dist.is_master_proc():
    #     return
    # Ensure that the checkpoint dir exists
    os.makedirs(cfg.EPISODE_DIR, exist_ok=True)

    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "cfg": cfg.dump(),
    }
    global _NAME_PREFIX
    _NAME_PREFIX = info + '_' + _NAME_PREFIX

    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch, cfg.EPISODE_DIR)
    torch.save(checkpoint, checkpoint_file)
    # print("Model checkpoint saved at path: {}".format(checkpoint_file))

    _NAME_PREFIX = 'model_epoch_'
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"]) if optimizer else ()
    return model


def delete_checkpoints(checkpoint_dir=None, keep="all"):
    """Deletes unneeded checkpoints, keep can be "all", "last", or "none"."""
    assert keep in ["all", "last", "none"], "Invalid keep setting: {}".format(keep)
    checkpoint_dir = checkpoint_dir if checkpoint_dir else get_checkpoint_dir()
    if keep == "all" or not os.path.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    checkpoints = sorted(checkpoints)[:-1] if keep == "last" else checkpoints
    [os.remove(os.path.join(checkpoint_dir, checkpoint)) for checkpoint in checkpoints]
    return len(checkpoints)
