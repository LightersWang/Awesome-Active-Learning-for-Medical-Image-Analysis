import os
import os.path as osp
import time
import argparse

def get_args(mode):
    # Training configurations
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--exp_comment', type=str, help='comment of current experiments')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)   
    parser.add_argument('--device', type=str, default='cuda:0', help='compute device',
                        choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--data_parallel', action='store_true', default=False, 
                        help='whether enable data parallel to use both cuda cards')

    # Model Options
    parser.add_argument('-m', '--model', type=str, default='unet_plain', help='model name',
                        choices=['unet_plain', ])
    parser.add_argument('--input_dimension', type=int, default=2, choices=[2, 3], 
                        help='spatial dimension of input & network')
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--unet_channels', type=int, nargs='+', default=[32, 64, 128, 256, 512],
                        help='#channels of every level in u-net, last is bottleneck dim')
    parser.add_argument('--normalization', type=str, default='batch', choices=['instance', 'batch'],
                        help='type of normalization in unet conv block')
    parser.add_argument('--dropout_prob', type=float, default=0.1, 
                        help='dropout probability for all conv blocks in unet')
    parser.add_argument('--deep_supervision', action='store_true', default=False, 
                        help='whether use deep supervision in unet')
    parser.add_argument('--deep_supervision_layer', type=int, default=3,
                        help='last x layers for deep supervision')
    
    # dataset
    parser.add_argument('--dataset',  help='dataset name')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes in dataset')
    parser.add_argument('--ignore_index', type=int, default=255, help='ignore index')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--total_itrs', type=int, default=30000, help='number of total iterations')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')    
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgdm', 'adam', 'adamw'],
                        help='choice of optimizer (default: adam)')
    parser.add_argument('--lr_schedule', type=str, default='poly', choices=['poly', 'none'],
                        help='choice of lr schedule (default: poly decay)')
    parser.add_argument('--grad_norm', type=float, default=10.0, help='gradient norm for clip')    

    # evaluation related
    parser.add_argument('--val_mode', type=str, default='slice', choices=['slice', 'volume'],
                        help='calculate slice-wise or volume-wise dice')
    parser.add_argument('--val_period', type=int, default=1000, help='validation frequency')
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size for validation')

    if 'active' in mode:
        parser.add_argument('--active_method', default='random', help='active learning query strategy')
        parser.add_argument('--datalist_path', type=str, default=None,
                            help='Load datalist files (to continue the experiment).')
        parser.add_argument('--init_iteration', type=int, default=0,
                            help='Initial active learning iteration (default: 0)')
        parser.add_argument('--max_iterations', type=int, default=5,
                            help='Number of active learning iterations (default: 5)')
        parser.add_argument('--percent_per_round', type=int, default=1,
                            help='percentage of whole dataset for every active selection round')
        parser.add_argument('--count_per_round', type=int, default=None,
                            help='count of selected samples for every active selection round')
        parser.add_argument('--init_datalist', type=str, default=None,
                            help='datalist path of the inital labeled pool')
        parser.add_argument('--init_checkpoint', type=str, default=None,
                            help='Load init checkpoint file to skip the initial iteration.')
        parser.add_argument('--coreset_distance', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                            help='Distance metircs for Core-Set.')

    args = parser.parse_args()
    args.mode = mode

    exp_name = [args.exp_dir, args.model, f'lr{args.lr}']

    time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())
    if 'active' in mode:
        exp_name.append(args.active_method)
        if args.exp_comment is not None:
            exp_name.append(args.exp_comment)
        exp_name = '_'.join(exp_name)
        seed_dir = '_'.join([f'seed{args.seed}', time_stamp])
        args.exp_dir = osp.join(exp_name, seed_dir)
    else:
        if args.exp_comment is not None:
            exp_name.append(args.exp_comment)
        exp_name.append(time_stamp)
        args.exp_dir = '_'.join(exp_name)
    
    return args
