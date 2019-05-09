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

    def __init__(self, model_path, prototxt_path):
        self.model_path = model_path
        self.prototxt_path = prototxt_path
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

    def detection_run(self, image):
        self.blob = cv2.dnn.blobFromImage(image, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), True)



if __name__ == '__main__':

    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\opencv\\'

    model_path = models_path + 'MobileNetSSD_deploy.caffemodel'
    prototxt_path = models_path + 'MobileNetSSD_deploy.prototxt.txt'

    object_test = OpenCV__Detection(model_path, prototxt_path)