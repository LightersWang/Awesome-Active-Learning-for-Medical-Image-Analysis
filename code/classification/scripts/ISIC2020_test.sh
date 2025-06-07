#!/usr/bin/env bash

python tools/test_model.py \
    --cfg configs/ISIC2020/evaluate/RESNET18.yaml \
    --ckpt paths/to/your/method/dir
    