# pedestrian-tracking

Track pedestrians from video or camera using OpenCV.

It detects people bounding boxes using a custom yolov4-tiny model and tracks them.

## Instructions

## Person Detector

The initial model was the first 29 layers of Yolov4-tiny pre-trained on the COCO dataset on 80 classes.

Then it was trained / fine-tuned just to detect people with a 416x416 resolution for 6000 iterations on the [CrowdHuman]( http://www.crowdhuman.org/) dataset improving the mean Average Precision (mAP) from 34.82% to 52.41%, and further to 61% increasing the resolution to 608x608 for inference.

## Tracker

## Thanks

To every contributor / researcher that has made possible the following:

+ https://github.com/AlexeyAB/darknet#pre-trained-models
+ https://github.com/alaksana96/darknet-crowdhuman
+ http://www.crowdhuman.org/
