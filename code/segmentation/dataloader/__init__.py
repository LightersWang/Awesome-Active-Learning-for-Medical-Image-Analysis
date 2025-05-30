from dataloader.acdc import ACDC
from dataloader.active_dataset import ActiveDataset


dataset_dict = {

    'ACDC': {
        'root': './data/acdc',
        'list': {
            'train': "./dataloader/datalist/acdc/train_slices.txt",
            'val'  : "./dataloader/datalist/acdc/val.txt",
            'test' : "./dataloader/datalist/acdc/test.txt",
        },
        'class':{
            'image': ACDC,
        }
    }, 
}


def get_dataset(args, split='train', region=False, empty_list=False, **kwargs):
    """Obtain a specified dataset class.
    Args:
        name (str): the name of datasets.
        split (str):
        region (boolean): region dataset or not
        empty_list (boolean): empty datalist for pool set
    """
    name = args.dataset
    assert name in dataset_dict, "Invalid dataset name!"

    root = dataset_dict[name]['root']

    # active spilt use train list
    if split in ["active-label", "active-pool"]:
        split4dict = 'train'
    else:
        split4dict = split

    # whether return a empty dataset, used for tgt pool dataset
    if empty_list:
        datalist = None
    else:
        datalist = dataset_dict[name]['list'][split4dict] 

    # return a image or region dataset
    if region:
        DATASET = dataset_dict[name]['class']['region']
        region_dict = dataset_dict[name]['region_dict'][split4dict]
        dataset = DATASET(args, root=root, datalist=datalist, split=split, region_dict=region_dict, **kwargs)
    else:
        DATASET = dataset_dict[name]['class']['image']
        dataset = DATASET(args, root=root, datalist=datalist, split=split, **kwargs)

    return dataset


def get_active_dataset(args):
    label_dataset = get_dataset(args, split='active-label', empty_list=True,  region=False)
    pool_dataset  = get_dataset(args, split='active-pool',  empty_list=False, region=False)
    dataset = ActiveDataset(args, pool_dataset, label_dataset)

    return dataset
