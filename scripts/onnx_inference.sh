#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python demo/ONNXRuntime/onnx_inference.py \
--model checkpoints/yolox_nano_breast_roi.onnx \
--image_path /media/ubuntu/HD/Data/BreastTidy/VindrMammo/images/P4882/P4882_CC_LEFT_f54a.png \
--output_dir YOLOX_outputs/yolox_nano/onnxruntime \
--score_thr 0.8 \
--input_shape "416,416"

