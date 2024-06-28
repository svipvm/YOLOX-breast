#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python ./tools/train.py \
--experiment-name Breast-ROI \
--exp_file exps/breast/yolox_nano.py \
--ckpt checkpoints/yolox_nano.pth \
--logger tensorboard \
--batch-size 64 \
--devices 0 \
--occupy 

