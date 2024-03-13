# Deep Active Learning Toolkit for Medical Image Segmentation in PyTorch

This is a code base for deep active learning for medical image segmentation written in [PyTorch](https://pytorch.org/). Our code is built on [D2ADA](https://github.com/tsunghan-wu/D2ADA) with heavy modifications.

## ACDC Dataset

Download the processed ACDC dataset in [Google Drive](https://drive.google.com/file/d/158vDssHPfYFuaPSMWfr5n6lsHMwvIYgf/view?usp=sharing)

## Execution Commands
### Active Learning on ACDC

```
bash segmentation/scripts/ACDC_al.sh
```

### Fully Supervised Learning on ACDC

```
bash segmentation/scripts/ACDC_sup.sh
```


## Active Learning Methods Supported
  * Least Confidence
  * Min-Margin
  * Max-Entropy
  * Coreset (L2 & Cosine distances) [1]
  * Batch Active learning with Diverse Gradient Embeddings (BADGE) [2]


## References

[1] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[2] Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., & Agarwal, A. (2020). Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR.


## Acknowledgement
- https://github.com/tsunghan-wu/D2ADA
- https://github.com/ej0cl6/deep-active-learning
- https://github.com/JordanAsh/badge