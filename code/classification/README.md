# Deep Active Learning Toolkit for Medical Image Classification in PyTorch

This is a code base for deep active learning for image classification written in [PyTorch](https://pytorch.org/). It is build on top of FAIR's [pycls](https://github.com/facebookresearch/pycls/). Our code is modified based on [deep-active-learning-pytorch](https://github.com/acl21/deep-active-learning-pytorch).

## Using the Toolkit

Please see [`GETTING_STARTED`](GETTING_STARTED.md) for brief instructions on installation, adding new datasets, basic usage examples, etc.

## Active Learning Methods Supported
  * Least Confidence
  * Min-Margin
  * Max-Entropy
  * Deep Bayesian Active Learning (DBAL) [1]
  * Bayesian Active Learning by Disagreement (BALD) [1]
  * Coreset (L2 & Cosine distances) [2]
  * Batch Active learning with Diverse Gradient Embeddings (BADGE) [3]


## Medical Datasets Supported
* [NCT-CRC-HE-100K](https://zenodo.org/records/1214456)
* [ISIC 2020](http://yann.lecun.com/exdb/mnist/)

The data loading of ISIC dataset is slow due to its high resolution. We recommend to resize each image to 300x300 offline for faster speed. Therefore, the data dir of ISIC 2020 is renamed as `ISIC_2020_Training_JPEG_300x300`.

Place the two datasets as follow:
```
code/classification/data/
├── ISIC2020
│   ├── ISIC_2020_Training_JPEG_300x300
│   ├── train
│   │   ├── ISIC_0015719.jpg
│   │   ...
│   │   └── ISIC_9999806.jpg
│   ├── test_split.csv
│   └── trainval_split.csv
└── NCT
    ├── CRC-VAL-HE-7K
    │   └── ...
    └── NCT-CRC-HE-100K
        └── ...
```

Follow the instructions in [`GETTING_STARTED`](GETTING_STARTED.md) to add a new dataset. 

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## References

[1] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1183–1192. JMLR. org, 2017.

[2] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[3] Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., & Agarwal, A. (2020). Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR.