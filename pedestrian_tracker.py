#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
parser.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file. Write only the name, without extension.")
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
 
# Open video capture, if no input file open camera with index 0
if args.input == "":
    args.input = 0
video_capture = cv2.VideoCapture(args.input)

# Open video write if there is output file
save_video = args.output != ""
if save_video:
    frameRate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frameRate,
        (frame_width, frame_height),
    )

# Process each frame
while video_capture.isOpened():
    read_ok, frame = video_capture.read()
    if not read_ok:
        break
 
    # Detecting objects
    classIds, scores, boxes = model.detect(frame, confThreshold=args.confidence, nmsThreshold=args.nms)
    
    # Computing tracks
    # TODO
 
    # Drawing information on the screen
    font = cv2.FONT_HERSHEY_DUPLEX
    for (class_id, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        confidence_label = str(int(score * 100))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label + ": " + confidence_label, (x, y - 5), font, 1, color, 2)
 
    # Show and write to output
    cv2.imshow("Pedestrian Tracker", frame)
    if save_video:
        video_writer.write(frame)
    
    # Close video window by pressing Escape
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
video_capture.release()
if save_video:
    video_writer.release()
