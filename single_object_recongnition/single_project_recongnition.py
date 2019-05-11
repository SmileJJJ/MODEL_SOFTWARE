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
from myLogger import LogHelper


class objects_recongnition():

    def __init__(self, require_method, program_method, logger):
        self.require_method = require_method
        self.program_method = program_method
        self.logger = logger
        self.models_path = '..\\..\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
        self.image_path = '..\\..\\MODEL_SOFTWARE\\resources\\image_test\\'

        #  转换data_convert的参数获取方式
        self.


    def run(self):
        if self.program_method == 'Program_ImageAI':
            self.Program_ImageAI_run()
        elif self.program_method == 'Program_ImageNet':
            self.Program_ImageNet_run()

    def Program_ImageAI_run(self):
        from Program_ImageAI.prediction_imageAI import Image_AI_Objects_Prediction
        object_test = Image_AI_Objects_Prediction(self.models_path, self.logger, predict_model='DenseNet121')
        object_test.predict_run(image_path=self.image_path + "test5.png")

    def Program_ImageNet_run(self):
        # imagenet模型训练模块
        pass


if __name__ == '__main__':

    logger = LogHelper('single_recongnition_log')

    require_method = 'single_object_recongnition'
    program_method = 'Program_ImageNet'

    object_recongnition = objects_recongnition(require_method, program_method, logger)
    object_recongnition.run()
