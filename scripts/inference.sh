#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python tools/demo.py image \
-f exps/breast/yolox_nano.py \
-c YOLOX_outputs/Breast-ROI/best_ckpt.pth \
--path /media/ubuntu/HD/Data/BreastTidy/INBreast/images/10194481 \
--conf 0.8 \
--nms 0.9 \
--tsize 416 \
--save_result \
--device gpu
