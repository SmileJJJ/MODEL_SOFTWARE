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
from myLogger import LogHelper


class objects_recongnition():

    def __init__(self, require_method, program_method, logger):
        self.require_method = require_method
        self.program_method = program_method
        self.logger = logger

        self.imageai_models_path = '..\\..\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
        self.opencv_model_path = '..\\..\\MODEL_SOFTWARE\\resources\\models\\opencv\\MobileNetSSD_deploy.caffemodel'
        self.opencv_prototxt_path = '..\\..\\MODEL_SOFTWARE\\resources\\models\\opencv\\MobileNetSSD_deploy.prototxt.txt'

        self.image_path = '..\\..\\MODEL_SOFTWARE\\resources\\image_test\\test1.png'
        self.new_image_path = '..\\..\\MODEL_SOFTWARE\\resources\\image_test\\test1_new.png'

        self.video_path = '..\\..\\MODEL_SOFTWARE\\resources\\video_test\\traffic.mp4'
        self.new_video_path = '..\\..\\MODEL_SOFTWARE\\resources\\video_test\\traffic_new.png'

    def run(self):
        if self.program_method == 'Program_ImageAI':
            self.Program_ImageAI_run()
        elif self.program_method == 'Program_OpenCV':
            self.Program_OpenCV_run()

    def Program_ImageAI_run(self):
        from Program_ImageAI.detection_imageAI import Image_AI_Objects_Detection
        object_test = Image_AI_Objects_Detection(self.imageai_models_path, self.logger, detect_target='image')
        object_test.detect_run(resource_path=self.image_path, save_path=self.new_image_path)

    def Program_OpenCV_run(self):
        from Program_OpenCV.detection_openCV import OpenCV_Detection
        object_test = OpenCV_Detection(self.opencv_model_path, self.opencv_prototxt_path, self.logger)
        try:
            vc = cv2.VideoCapture(self.video_path)
            self.logger.writeLog('detection running', level='info')
        except:
            self.logger.writeLog('video has not found', level='error')
        while True:
            frame = vc.read()[1]
            try:
                image = object_test.detection_run(frame)
            except:
                self.logger.writeLog('video has something wrong,please check it', level='error')
                break
            cv2.imshow('frame', image)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        self.logger.writeLog('detection end, program quit ', level='info')


if __name__ == '__main__':
    logger = LogHelper('objects_recongnition_log')

    require_method = 'scene_objects_recongnition'
    program_method = 'Program_ImageAI'

    object_recongnition = objects_recongnition(require_method, program_method, logger)
    object_recongnition.run()
