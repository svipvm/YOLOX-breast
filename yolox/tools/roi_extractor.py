import os, sys, cv2
import numpy as np

import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, vis, mkdir

COCO_CLASSES = ['breast']

def extract_roi_otsu(img, gkernel=(5, 5)):
    """WARNING: this function modify input image inplace."""
    ori_h, ori_w = img.shape[-2:]
    # clip percentile: implant, white lines
    if len(img.shape) == 3:
        img = img[0]
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # upper = np.percentile(img, 95)
    # img[img > upper] = np.min(img)
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    img_bin = cv2.dilate(img_bin, element, iterations=3)
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("YOLOX_outputs/yolox_nano/inference/temp.png", img_bin)
    
    if len(cnts) == 0:
        return None, None
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    x0, y0, w, h = cv2.boundingRect(cnt)
    
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    
    return [x0, y0, x1, y1], area_pct


class ROIExtractor:
    def __init__(self,
                 onnx_path,
                 input_size=(416, 416),
                 conf_thr=0.5,
                 nms_thr=0.9,
                 area_pct_thres=0.04):
        self.input_size = input_size
        self.input_h, self.input_w = input_size
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.area_pct_thres = area_pct_thres

        self.model = onnxruntime.InferenceSession(onnx_path,
            providers=['CUDAExecutionProvider'])

    def __call__(self, **kwargs):
        return self.inference(**kwargs)
    
    def inference(self, inp):
        ori_h, ori_w = inp.shape[:2]
        img, ratio = preprocess(inp, self.input_size)

        ort_inputs = {self.model.get_inputs()[0].name: img[None, :, :, :]}
        output = self.model.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_size)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(
            boxes_xyxy, scores, nms_thr=self.nms_thr, score_thr=self.conf_thr)

        if dets is not None:
            # select box with highest confident
            dets = np.array(dets)
            det = dets[dets[:, 4].argmax()]
            pre_class = COCO_CLASSES[int(det[5])]
            x0 = min(max(int(det[0]), 0), ori_w)
            y0 = min(max(int(det[1]), 0), ori_h)
            x1 = min(max(int(det[2]), 0), ori_w)
            y1 = min(max(int(det[3]), 0), ori_h)
            area_pct = (x1 - x0) * (y1 - y0) / (ori_h * ori_w)
            if area_pct >= self.area_pct_thres:
                return {
                    "box": [x0, y0, x1, y1], 
                    "area_pct": area_pct, 
                    "confident": np.around(det[4], 2), 
                    "class": pre_class
                }

        # if YOLOX fail, try Otsu thresholding + find contours
        img, ratio = preprocess(inp, self.input_size, pad_mode='min')
        xyxy, area_pct = extract_roi_otsu(img)
        # if both fail, use full frame
        if xyxy is not None:
            if area_pct >= self.area_pct_thres:
                # print('ROI detection: using Otsu.')
                x0, y0, x1, y1 = xyxy
                x0 = min(max(int(x0 / ratio), 0), ori_w)
                y0 = min(max(int(y0 / ratio), 0), ori_h)
                x1 = min(max(int(x1 / ratio), 0), ori_w)
                y1 = min(max(int(y1 / ratio), 0), ori_h)
                return {
                    "box": [x0, y0, x1, y1], 
                    "area_pct": area_pct, 
                    "confident": None, 
                    "class": None
                }
        # print('ROI detection: both fail.')
        return {
            "box": [0, 0, ori_w, ori_h], 
            "area_pct": None, 
            "confident": None, 
            "class": None
        }

if __name__ == '__main__':
    roi_extractor = ROIExtractor(
        'checkpoints/yolox_nano_breast_roi.onnx', (416, 416), 0.5, 0.9)
    
    img_file = "/media/ubuntu/HD/Data/BreastTidy/VindrMammo/images/P4882/P4882_CC_LEFT_f54a.png"
    img = cv2.imread(img_file)
    result = roi_extractor.inference(img)

    origin_img = vis(img, [result['box']], [1.0], [0], class_names=COCO_CLASSES)
    # origin_img = vis(img, [[50, 50, 500, 500]], [1.0], [0], class_names=COCO_CLASSES)

    output_dir = 'YOLOX_outputs/yolox_nano/inference'
    mkdir(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(img_file))
    cv2.imwrite(output_path, origin_img)

    print(result)
