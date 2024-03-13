from active_selection.badge import BADGESelector
from active_selection.core_set import CoreSetSelector
from active_selection.random_selection import RandomSelector
from active_selection.softmax_uncertainty import SoftmaxUncertaintySelector


def get_active_selector(args, logger):
    return image_active_selector(args, logger)


def image_active_selector(args, logger):
    active_method = args.active_method
    batch_size = args.val_batch_size
    num_workers = args.num_workers

    if active_method == 'random':
        return RandomSelector()
    elif active_method == 'softmax_confidence':
        return SoftmaxUncertaintySelector(args, logger, batch_size, num_workers, 'softmax_confidence')
    elif active_method == 'softmax_margin':
        return SoftmaxUncertaintySelector(args, logger, batch_size, num_workers, 'softmax_margin')
    elif active_method == 'softmax_entropy':
        return SoftmaxUncertaintySelector(args, logger, batch_size, num_workers, 'softmax_entropy')
    elif active_method == 'core_set_L2':
        args.coreset_distance = 'euclidean'
        return CoreSetSelector(args, logger, batch_size, num_workers)
    elif active_method == 'core_set_cosine':
        args.coreset_distance = 'cosine'
        return CoreSetSelector(args, logger, batch_size, num_workers)
    elif active_method == 'badge':
        return BADGESelector(args, logger, 1, num_workers, multiple_loss='add')
    else:
        raise NotImplementedError(f"{active_method} is not supported for image-based ADA")

