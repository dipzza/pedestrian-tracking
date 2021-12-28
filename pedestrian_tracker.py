#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import time
from sort import Sort

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="./data/test.mp4",
	help="path to (optional) input video file")
parser.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
parser.add_argument("-s", "--show", action='store_true',
	help="Show video while processing")
parser.add_argument("-ev", "--evaluate", type=str, default="",
	help="path to output results on MOT format")
parser.add_argument("-c", "--confidence", type=float, default=0.35,
	help="minimum probability for detections")
parser.add_argument("--nms", type=float, default=0.4,
	help="threshold for non maxima supression")
parser.add_argument("--max_age", type=int, default=1,
    help="Maximum number of frames to keep alive a track without associated detections.")
parser.add_argument("--min_hits", type=int, default=3,
    help="Minimum number of associated detections before track is initialised.")
parser.add_argument("--iou_threshold", type=float, default=0.3, 
                    help="Minimum IOU for match.")
args = parser.parse_args()

# Defining network paths, classes and resolution
yolo_weight = "model/yolov4-tiny-custom.weights"
yolo_config = "model/yolov4-tiny-custom.cfg"
res = 608 #Original 416. Increase in 32 increments for better but slower detection

# Load Yolo model and set backend
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weight)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(res, res), swapRB=True)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  #CPUs
#model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) #Nvidia GPUs
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL) #AMD/Intel GPUs

# Create SORT MOT tracker
mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
 
# Open video capture, if no input file open camera with index 0
if args.input == "":
    args.input = 0
if args.evaluate:
    video_capture = cv2.VideoCapture(args.input + '%06d.jpg', cv2.CAP_IMAGES)
else:
    video_capture = cv2.VideoCapture(args.input)

# Open video write if there is output file
if args.output:
    frameRate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frameRate,
        (frame_width, frame_height),
    )

frame_n = 0
# Process each frame
start_time = time.time()
while video_capture.isOpened():
    read_ok, frame = video_capture.read()
    if not read_ok:
        break
    frame_n += 1

    # Detecting objects
    _, scores, boxes = model.detect(frame, confThreshold=args.confidence, nmsThreshold=args.nms)
    
    # Compute tracks
    if len(boxes) > 0:
        # Change x, y, w, h to x1, y1, x2, y2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        detections = np.c_[boxes, scores]
        tracks_boxes_ids = mot_tracker.update(detections)
    else:
        tracks_boxes_ids = mot_tracker.update()
     
    # Drawing tracks on screen
    if args.output or args.show:
        tracks_boxes_ids_int = tracks_boxes_ids.astype('int32')
        font = cv2.FONT_HERSHEY_DUPLEX
        for track_box_id in tracks_boxes_ids_int:
            x1, y1, x2, y2 = track_box_id[:4]
            tr_id = str(track_box_id[4])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "ID: " + tr_id, (x1, y1 - 5), font, 1, color, 2)
    
    # Write MOT format txt
    if args.evaluate:
        with open(args.evaluate, 'a') as file:
            for track_box_id in tracks_boxes_ids:
                x1, y1, x2, y2 = track_box_id[:4]
                new_line = [frame_n, track_box_id[4], x1, y1, x2-x1, y2-y1, -1, -1, -1, -1]
                new_line = ', '.join([str(x) for x in new_line])
                file.write(new_line + '\n')
    
    # Show/write frame
    if args.show:
        cv2.imshow("Pedestrian Tracker", frame)
        if cv2.waitKey(1) == 27:
            break
    if args.output:
        video_writer.write(frame)

total_time = time.time() - start_time
print("FPS: " + str(frame_n / total_time))

cv2.destroyAllWindows()
video_capture.release()
if args.output:
    video_writer.release()
