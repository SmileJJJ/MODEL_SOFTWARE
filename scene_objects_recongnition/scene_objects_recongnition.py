#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = '场景多个物体检测'
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
import cv2
import os
import sys
import numpy as np 

if __name__ == '__main__':

    imageai_models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
    opencv_model_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\opencv\\MobileNetSSD_deploy.caffemodel'
    opencv_prototxt_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\opencv\\MobileNetSSD_deploy.prototxt.txt'

    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\test1.png'
    new_image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\test1_new.png'

    video_path = 'F:\\MODEL_SOFTWARE\\resources\\video_test\\traffic.mp4'
    new_video_path = 'F:\\MODEL_SOFTWARE\\resources\\video_test\\traffic_new.png'



    # detect_method = 'ImageAI'
    detect_method = 'OpenCV'

    if detect_method == 'ImageAI':
        from Program_ImageAI.detection_imageAI import Image_AI_Objects_Detection
        object_test = Image_AI_Objects_Detection(imageai_models_path, detect_target='video')
        # object_test.detect_run(resource_path=image_path, save_path=new_image_path)
        object_test.detect_run(resource_path=video_path, save_path=new_video_path)

    if detect_method == 'OpenCV':
        from Program_OpenCV.detection_openCV import OpenCV__Detection
        object_test = OpenCV__Detection(opencv_model_path, opencv_prototxt_path)
        vc = cv2.VideoCapture(video_path)
        while True:
            frame = vc.read()[1]
            image = object_test.detection_run(frame)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
