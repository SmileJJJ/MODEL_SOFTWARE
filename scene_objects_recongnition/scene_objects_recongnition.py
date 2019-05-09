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
import os
import sys

if __name__ == '__main__':

    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\imageai\\'

    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\test1.png'
    new_image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\test1_new.png'

    video_path = 'F:\\MODEL_SOFTWARE\\resources\\video_test\\traffic.mp4'
    new_video_path = 'F:\\MODEL_SOFTWARE\\resources\\video_test\\traffic_new.png'

    detect_method = 'ImageAI'
    if detect_method == 'ImageAI':
        from Program_ImageAI.detection_imageAI import Image_AI_Objects_Detection
        object_test = Image_AI_Objects_Detection(models_path, detect_target='video')
        # object_test.detect_run(resource_path=image_path, save_path=new_image_path)
        object_test.detect_run(resource_path=video_path, save_path=new_video_path)

