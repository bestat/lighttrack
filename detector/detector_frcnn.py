# import necessary libraries
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cuda = torch.cuda.is_available()

if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index


def inference_frcnn(img_path):
    img = Image.open(img_path)
    return inference_frcnn_from_img(img)

def inference_frcnn_from_img(img):
    human_candidates = get_prediction(img)
    return human_candidates

def get_prediction(img, threshold=0.80):
    """
    get_prediction
    parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
    method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.

    """
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    if cuda:
        img = img.cuda()
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    human_candidates = []

    for i in range(len(pred_boxes)):
        if pred_class[i] == "person":
            x1, y1 = pred_boxes[i][0]
            x2, y2 = pred_boxes[i][1]
            box_w = x2 - x1
            box_h = y2 - y1
            human_candidates.append([x1, y1, box_w, box_h])

    return human_candidates


if __name__ == "__main__":
    img_path = "./detector/people.jpg"
    human_candidates = inference_frcnn(img_path)
    print("human_candidates:", human_candidates)
