#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="path to input image file", required=True)
parser.add_argument("-o", "--output", type=str, default="output.png",
	help="path to (optional) output image file")
parser.add_argument("-c", "--confidence", type=float, default=0.35,
	help="minimum probability for detections")
parser.add_argument("-n", "--nms", type=float, default=0.4,
	help="threshold for non maxima supression")
args = parser.parse_args()

# Defining network paths, classes and resolution
yolo_weight = "model/yolov4-tiny-custom.weights"
yolo_config = "model/yolov4-tiny-custom.cfg"
label = "person"
res = 608

# Load Yolo model
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weight)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(res, res), swapRB=True)


# Process frame
frame = cv2.imread(args.input)
if frame is None:
    print("Error: Unable to read image file", args.input)
    exit(-1)
 
# Detecting objects
classIds, scores, boxes = model.detect(frame, confThreshold=args.confidence, nmsThreshold=args.nms)
 
# Drawing information on the screen
font = cv2.FONT_HERSHEY_DUPLEX
for (class_id, score, box) in zip(classIds, scores, boxes):
    x, y, w, h = box
    confidence_label = str(int(score * 100))
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label + ": " + confidence_label, (x, y - 5), font, 1, color, 2)
 
# Show and write to output
if (args.output != ""):
    cv2.imwrite(args.output, frame)
