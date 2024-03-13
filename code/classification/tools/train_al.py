import os
import sys
import time
import torch
import argparse
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from monai.utils import set_determinism

# local
for path in ['..', '.']:
    if path not in sys.path:
        sys.path.insert(0, path)

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', dest='exp_name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', dest='al', help='AL Method', required=True, type=str)
    parser.add_argument('--seed', dest='seed', help='Random Seed', required=True, type=int)
    parser.add_argument('--coreset-distance', dest='coreset_distance', help='Random Seed', choices=['L2', 'cosine'], type=str)

    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )

def is_better_metrics(best_val_metrics, val_set_metrics):
    # use first metric for model selection
    m = cfg.DATASET.METRICS[0]
    if m in ['Accuracy', 'AUC', 'F1']:
        return val_set_metrics[m] > best_val_metrics[m]
    else:
        raise NotImplementedError


def main(cfg):

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    loader_cfg = {"num_workers": cfg.DATA_LOADER.NUM_WORKERS, 
                  "pin_memory": cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)
        logger.info(f'Assigned random RNG_SEED: {cfg.RNG_SEED}')

    # Set determinism
    set_determinism(cfg.RNG_SEED)

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('.'), cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{cfg.EXP_NAME}/seed{xxx}_{timestamp}
    al_dir = os.path.join(dataset_out_dir, cfg.EXP_NAME)
    os.makedirs(al_dir, exist_ok=True)

    time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())
    if cfg.EXP_NAME == 'auto':
        exp_dir = time_stamp
    else:
        exp_dir = "_".join([f'seed{cfg.RNG_SEED}', time_stamp]) 
    exp_dir = os.path.join(al_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.".format(exp_dir))
        logger.info("Experiment Directory is {}.".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.".format(exp_dir))
        logger.info("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Initialize tensorboard
    writer = SummaryWriter(cfg.EXP_DIR)

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    logger.info('======== PREPARING DATA AND MODEL ========')
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('.'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}".format(cfg.DATASET.NAME, train_size, test_size))
    
    split_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, 'split')
    os.makedirs(split_dir, exist_ok=True)
    if cfg.ACTIVE_LEARNING.INIT_L > 1:
        cfg.ACTIVE_LEARNING.INIT_L /= train_size
    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(
        initial_labeled_ratio=cfg.ACTIVE_LEARNING.INIT_L, 
        val_split_ratio=cfg.DATASET.VAL_RATIO, 
        data=train_data, 
        seed_id=cfg.RNG_SEED, 
        save_dir=split_dir
    )

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(
        lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, 
        uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, 
        valSetPath=cfg.ACTIVE_LEARNING.VALSET_PATH
    )

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data, **loader_cfg)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TEST.BATCH_SIZE, data=train_data, **loader_cfg)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TEST.BATCH_SIZE, seed_id=cfg.RNG_SEED, **loader_cfg)

    # Initialize the model
    model = model_builder.build_model(cfg)
    print("model: {}\n".format(cfg.MODEL.TYPE))
    print(model)
    logger.info("model: {}".format(cfg.MODEL.TYPE))
    logger.info(str(model))
    cfg.ACTIVE_LEARNING.INIT_MODEL_DIR = os.path.join(
        dataset_out_dir, f'inital_model_ratio{cfg.ACTIVE_LEARNING.INIT_L}.pyth')

    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)
    print("optimizer: {}\n".format(optimizer))
    logger.info("optimizer: {}".format(optimizer))

    print("AL Query Method: {}\tMax AL Episodes: {}".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\tMax AL Episodes: {}".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        if cur_episode > 0:
            if not cfg.ACTIVE_LEARNING.LOAD_PREVIOUS_MODEL:
                model = model_builder.build_model(cfg)
                optimizer = optim.construct_optimizer(cfg, model)
                logger.info("Re-initialize the model weight in this active learning episode.")
            else:
                logger.info("Re-use the model weight in the last active learning episode.")
        
        print("======== EPISODE {} BEGINS ========".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        if (cur_episode == 0) and os.path.exists(cfg.ACTIVE_LEARNING.INIT_MODEL_DIR):
            # inital model has already trained! just load it
            logger.info(f"Found existed trained inital model {cfg.ACTIVE_LEARNING.INIT_MODEL_DIR}. Load it to skip training.")
            print(f"Found existed trained inital model {cfg.ACTIVE_LEARNING.INIT_MODEL_DIR}. Load it to skip training.")
            checkpoint_file = cfg.ACTIVE_LEARNING.INIT_MODEL_DIR
        else:
            # Train model
            print("======== TRAINING ========")
            logger.info("======== TRAINING ========")
            
            best_val_metrics, best_val_epoch, checkpoint_file = train_model(lSet_loader, valSet_loader, model, optimizer, cfg, writer, cur_episode)
            best_val_mname = cfg.DATASET.METRICS[0]
            best_val_m = best_val_metrics[best_val_mname]
            print(f"Best Validation {best_val_mname}: {round(best_val_m, 4)}\tBest Epoch: {best_val_epoch}\n")
            logger.info(f"EPISODE {cur_episode} Best Validation {best_val_mname}: {round(best_val_m, 4)}\tBest Epoch: {best_val_epoch}")
            
        # Test best model checkpoint
        print("======== TESTING ========\n")
        logger.info("======== TESTING ========")
        test_metrics = test_model(test_loader, checkpoint_file, cfg, writer, cur_episode)
        test_mname = cfg.DATASET.METRICS[0]
        test_m = test_metrics[test_mname]
        print(f"EPISODE {cur_episode} Test {test_mname} {test_m}.\n")
        logger.info(f"EPISODE {cur_episode} Test {test_mname} {test_m}.\n")
        
        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========")
        al_obj = ActiveLearning(data_obj, cfg)
        clf_model = model_builder.build_model(cfg)
        clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
        activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data)

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data, **loader_cfg)
        valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data, **loader_cfg)
        # uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data, **loader_cfg)

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        print("================================\n\n")
        logger.info("================================\n\n")


def train_model(train_loader, val_loader, model, optimizer, cfg, writer, cur_episode):

    start_epoch = 0
    loss_fun = losses.get_loss_fun()
    logger.info('Loss function: {}'.format(loss_fun))

    # Create meters
    train_meter = TrainMeter(len(train_loader), writer)
    val_meter = ValMeter(len(val_loader), writer)

    # Perform the training loop
    print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))

    # Best checkpoint model and optimizer states
    best_model_state, best_opt_state = None, None
    best_val_metrics = {m: 0. for m in cfg.DATASET.METRICS}
    best_val_epoch = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_loss = train_epoch(
            train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)
        
        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_metrics = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_loader.dataset.no_aug = False

            if is_better_metrics(best_val_metrics, val_set_metrics):
                best_val_metrics = deepcopy(val_set_metrics)
                best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()
                
                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            val_str = [
                f'Epoch: {cur_epoch+1}/{cfg.OPTIM.MAX_EPOCH}',
                f'Loss: {round(train_loss, 4)}']
            for m, val_m in best_val_metrics.items():
                val_str.append(f'{m.capitalize()}: {round(val_m, 4)}')
            val_str = "\t".join(val_str)
            print(val_str)

    if cur_episode > 0:
        # Save the best model checkpoint (Episode level)
        m_model_selection = cfg.DATASET.METRICS[0]
        checkpoint_file = cu.save_checkpoint(
            info=f"valBest_{m_model_selection}_" + str(int(best_val_metrics[m_model_selection])), 
            model_state=best_model_state, optimizer_state=best_opt_state, 
            epoch=best_val_epoch, cfg=cfg)
    else:
        # Save model trained on inital labeled dataset
        checkpoint = {
            "epoch": best_val_epoch,
            "model_state": best_model_state,
            "optimizer_state": best_opt_state,
            "cfg": cfg.dump(),
        }
        checkpoint_file = cfg.ACTIVE_LEARNING.INIT_MODEL_DIR
        torch.save(checkpoint, checkpoint_file)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}'.format(checkpoint_file))

    return best_val_metrics, best_val_epoch, checkpoint_file


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg):
    """Performs one epoch of training."""

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS > 1:  
        train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()

    # This basically notes the start time in timer class defined in utils/timer.py
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err = mu.topk_errors(preds, labels, [1,])[0]
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # if (cur_iter + 1) == 1 or (cur_iter + 1) % 5 == 0 or (cur_iter + 1) == len(train_loader):
        print('Training Epoch: {}/{}\tIter: {}/{}\tLoss: {}\tTop1 Error: {}'.format(
            cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter+1, len(train_loader), 
            round(loss, 5), round(top1_err, 5)))

        # Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(
            top1_err=top1_err, loss=loss, lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    return loss


def test_model(test_loader, checkpoint_file, cfg, writer, cur_episode):

    test_meter = TestMeter(len(test_loader), writer)

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
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
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
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()

    # metric calculation
    if 'Accuracy' in cfg.DATASET.METRICS:
        test_acc = 100. - (misclassifications / totalSamples)
        test_metrics.update({'Accuracy': test_acc})
    if ('AUC' in cfg.DATASET.METRICS) or ('F1' in cfg.DATASET.METRICS):
        test_probs = torch.cat(test_probs, dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

        if 'AUC' in cfg.DATASET.METRICS:
            test_auc = roc_auc_score(test_labels, test_probs[:, 1])
            test_metrics.update({'AUC': test_auc})
            
        if 'F1' in cfg.DATASET.METRICS:
            # find optimal threshold by maximizing f1 score
            precision, recall, thresh = precision_recall_curve(test_labels, test_probs[:, 1])
            fscore = (2 * precision * recall) / (precision + recall)
            thresh_opt = thresh[np.argmax(fscore)].astype(np.float64)
            test_pred = np.array(test_probs[:, 1] > thresh_opt, dtype=int)

            test_f1 = f1_score(test_labels, test_pred)
            test_metrics.update({'F1': test_f1})

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch, test_metrics)
    test_meter.reset()

    return test_metrics


if __name__ == "__main__":
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    cfg.RNG_SEED = argparser().parse_args().seed
    cfg.EXP_NAME = argparser().parse_args().exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = argparser().parse_args().al
    cfg.ACTIVE_LEARNING.CORESET_DISTANCE = argparser().parse_args().coreset_distance
    main(cfg)
