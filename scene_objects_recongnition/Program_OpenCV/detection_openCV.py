#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'None'
__author__ = 'None'
__mtime__ = 'None'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

from imutils.video import FPS
import numpy as np
import time
import cv2
import os

class OpenCV__Detection():

    def __init__(self, model_path, prototxt_path, CONFIDENCE=0.2):
        self.CONFIDENCE = CONFIDENCE
        self.model_path = model_path
        self.prototxt_path = prototxt_path
        # 加载用caffe训练好的模型，prototxt：模型结构文件，model：模型文件
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def detection_run(self, frame):
        (h, w) = frame.shape[:2]
        self.blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), True)
        self.net.setInput(self.blob)
        detections = self.net.forward()
        for i in np.arange(0, detections.shape[1]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.CONFIDENCE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startx, starty, endx, endy) = box.astype('int')

                label = '{}:{:.2f}%'.format(self.CLASSES[idx], confidence*100)
                cv2.rectangle(frame, (startx, starty), (endx, endy), self.COLORS[idx], 2)
                y = starty - 15 if starty - 15 > 15 else starty + 15
                cv2.putText(frame, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx, 2])
        return frame


if __name__ == '__main__':

    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\opencv\\'

    model_path = models_path + 'MobileNetSSD_deploy.caffemodel'
    prototxt_path = models_path + 'MobileNetSSD_deploy.prototxt.txt'

    object_test = OpenCV__Detection(model_path, prototxt_path)

    vc = cv2.VideoCapture(0)
    while True:
        frame = vc.read()[1]
        image = object_test.detection_run(frame)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()