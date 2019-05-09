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
import os

if __name__ == '__main__':

    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\'

    detect_method = 'ImageAI'
    if detect_method == 'ImageAI':
        from Program_ImageAI.prediction_imageAI import Image_AI_Objects_Prediction
        object_test = Image_AI_Objects_Prediction(models_path, predict_model='DenseNet121')
        object_test.predict_run(image_path=image_path + "test5.png")
