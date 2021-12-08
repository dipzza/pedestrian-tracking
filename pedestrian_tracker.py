#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

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

# Set up tracker.
# Instead of CSRT, you can also use
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
 
if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
         tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
 
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
    #-- beginnig of new code --
    # Start timer
    timer = cv2.getTickCount()
 
    # Update tracker    
    ok, bbox = tracker.update(frame)
 
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    # Display result
    #cv2.imshow("Tracking", frame)
    
    #-- end of new code --
 
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
