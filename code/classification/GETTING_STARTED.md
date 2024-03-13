# Getting Started

## Understanding the Config File
```
# Folder name where best model logs etc are saved. Setting EXP_NAME: "auto" creates a timestamp named folder
EXP_NAME: 'YOUR_EXPERIMENT_NAME'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
RNG_SEED: 1
# GPU ID you want to execute the process on (this feature isn't working as of now, use the commands shown in this file below instead)
GPU_ID: '3'
DATASET:
  NAME: CIFAR10 # or CIFAR100, MNIST, SVHN, TINYIMAGENET, IMBALANCED_CIFAR10/100
  ROOT_DIR: 'data' # Relative path where data should be downloaded
  # Specifies the proportion of data in train set that should be considered as the validation data
  VAL_RATIO: 0.1
  # Data augmentation methods - 'simclr', 'randaug', 'hflip'
  AUG_METHOD: 'hflip' 
MODEL:
  # Model type. 
  # Choose from vgg style ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]
  # or from resnet style ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
  # 'wide_resnet50_2', 'wide_resnet101_2']
  # or `alexnet`
  TYPE: resnet18
  NUM_CLASSES: 10
OPTIM:
  TYPE: 'sgd' # or 'adam'
  BASE_LR: 0.025
  # Learning rate policy select from {'cos', 'exp', 'steps' or 'none'}
  LR_POLICY: cos
  # Steps for 'steps' policy (in epochs)
  STEPS: [0] #[0, 30, 60, 90]
  # Training Epochs
  MAX_EPOCH: 1
  # Momentum
  MOMENTUM: 0.9
  # Nesterov Momentum
  NESTEROV: False
  # L2 regularization
  WEIGHT_DECAY: 0.0005
  # Exponential decay factor
  GAMMA: 0.1
TRAIN:
  SPLIT: train
  # Training mini-batch size
  BATCH_SIZE: 96
  # Image size
  IM_SIZE: 32
  IM_CHANNELS = 3
  # Evaluate model on test data every eval period epochs
  EVAL_PERIOD: 2
TEST:
  SPLIT: test
  # Testing mini-batch size
  BATCH_SIZE: 200
  # Image size
  IM_SIZE: 32
  # Saved model to use for testing (useful when running `tools/test_model.py`)
  MODEL_PATH: ''
DATA_LOADER:
  NUM_WORKERS: 4
CUDNN:
  BENCHMARK: True
ACTIVE_LEARNING:
  # Active sampling budget (at each episode)
  BUDGET_SIZE: 5000
  # Active sampling method
  SAMPLING_FN: 'dbal' # 'random', 'uncertainty', 'entropy', 'margin', 'bald', 'vaal', 'coreset', 'ensemble_var_R'
  # Initial labeled pool ratio (% of total train set that should be labeled before AL begins)
  INIT_L: 0.1
  # Max AL episodes
  MAX_ITER: 5
  DROPOUT_ITERATIONS: 25 # Used by DBAL
```

Please refer to `pycls/core/config.py` to configure your experiments at a deeper level. 


## Execution Commands
### Active Learning on NCT dataset

```
bash classification/scripts/NCT_al.sh
```

### Fully Supervised Learning on NCT dataset

```
bash classification/scripts/NCT_supervised.sh
```
### Active Learning on ISIC 2020 dataset

```
bash classification/scripts/ISIC2020_al.sh
```

### Fully Supervised Learning on ISIC 2020 dataset

```
bash classification/scripts/ISIC2020_supervised.sh
```



## Add Your Own Dataset 

To add your own dataset, you need to do the following: 
1. Write the PyTorch Dataset code for your custom dataset (or you could directly use the ones [PyTorch provides](https://pytorch.org/vision/stable/datasets.html)). 
2. Create a sub class of the above Dataset with some desirable modifications and add it to the `pycls/datasets/custom_datasets.py`.
    * We add two new variables to the dataset - a boolean flag `no_aug` and `test_transform`. 
    * We set the flag `no_aug = True` before iterating through unlabeled and the validations dataloaders so that data doesn't get augmented. 
    * See how we modify the `__get_item__` function to achieve that:
```
class CIFAR10(torchvision.datasets.CIFAR10):
      def __init__(self, root, train, transform, test_transform, download=True):
          super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
          self.test_transform = test_transform
          self.no_aug = False
  
      def __getitem__(self, index: int):
          """
          Args:
              index (int): Index
  
          Returns:
              tuple: (image, target) where target is index of the target class.
          """
          img, target = self.data[index], self.targets[index]
  
          # doing this so that it is consistent with all other datasets
          # to return a PIL Image
          img = Image.fromarray(img)
          
          ##########################
          # set True before iterating through unlabeled or validation set
          if self.no_aug: 
              if self.test_transform is not None:
                  img = self.test_transform(img)            
          else:
              if self.transform is not None:
                  img = self.transform(img)
          #########################
          
          return img, target
```
3. Add your dataset in `pycls/dataset/data.py` 
    * Add appropriate preprocessing steps to `getPreprocessOps` 
    * Add the dataset call to `getDataset`
4. Create appropriate config `yaml` files and use them for training AL.


## Some Comments About Our Toolkit
* Our toolkit currently only supports 'SGD' (with learning rate scheduler)  and 'Adam' (no scheduler). 
* We log everything. Our toolkit saves the indices of the initial labeled pool, samples queried each episode, episode wise best model, visual plots for "Iteration vs Loss", "Epoch vs Val Accuracy", "Episode vs Test Accuracy" and more. Please check an experiment's logs at `output/CIFAR10/resnet18/ENT_1/` for clarity.
* We added dropout (p=0.5) to all our models just before the final fully connected layer. We do this to allow the DBAL and BALD query methods to work.
* We also provide an iPython notebook that aggregates results directly from the experiment folders. You can find it at `output/results_aggregator.ipynb`. 
* If you add your own dataset, please make sure you to create the custom version as explained in point 2 in the instructions. Failing to do that would mean that your unlabeled data (big red flag for AL) and validation data will have been augmentated. This is because we use a single dataset instance and subset and index based dataloaders.   
* We tested the toolkit only on a Linux machine with Python 3.8.
* Please create an issue with appropriate details:
  * if you are unable to get the toolkit to work or run into any problems
  * if we have not provided credits correctly to the rightful owner (please attach proof)
  * if you notice any flaws in the implementation
