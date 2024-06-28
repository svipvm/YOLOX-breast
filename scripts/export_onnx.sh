#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python tools/export_onnx.py \
--output-name checkpoints/yolox_nano_breast_roi.onnx \
-f exps/breast/yolox_nano.py \
-c YOLOX_outputs/Breast-ROI/best_ckpt.pth \
--dynamic
