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

if __name__ == '__main__':

    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\'
    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\'

    detect_method = 'ImageAI'
    if detect_method == 'ImageAI':
        from Program_ImageAI.detection_imageAI import Image_AI_Objects_Detection
        object_test = Image_AI_Objects_Detection(models_path + "resnet50_coco_best_v2.0.1.h5")
        object_test.detect_run(image_path=image_path + "test1.png",
                               save_path=image_path + "test1_new.png")