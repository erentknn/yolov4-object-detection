"""
├── yolo
│   ├── labels.txt
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
├── people.jpg
├── people_out.jpg
├── street.jpg
├── street_out.jpg
├── video.mp4
├── video_out.avi
├── yolo_image.py
└── yolo_video.py
if program cant find yolo folder in main folder it will crash."""
# example usage: python yolo_video.py -i video.mp4 -o video_out.avi
# if you want to resize video, uncomment line 110
import argparse
import glob
import time

import cv2
import imutils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
                    help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
parser.add_argument("-d", "--display", type=int, default=1,
                    help="display output or not (1/0)")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="confidence threshold")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
                    help="non-maximum supression threshold")

args = vars(parser.parse_args())

CONFIDENCE_THRESHOLD = args["confidence"]
NMS_THRESHOLD = args["threshold"]
vc = cv2.VideoCapture(args["input"])
weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

class_names = list()
with open(labels, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
writer = None

def detect_people(frm, net, ln):
    (H, W) = frm.shape[:2]
    blob = cv2.dnn.blobFromImage(frm, 1/255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(ln)
    end_time = time.time()

    boxes = []
    classIds = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIds[i]]]
            cv2.rectangle(frm, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(class_names[classIds[i]], confidences[i])
            cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
            cv2.putText(frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    #frame = imutils.resize(frame, 700)
    detect_people(frame, net, layer)
    
    if args["display"] == 1:
        cv2.imshow("detections", frame)

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)
