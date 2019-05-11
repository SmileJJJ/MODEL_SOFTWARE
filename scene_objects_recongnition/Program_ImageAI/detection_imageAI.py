#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'imageai_objection物体检测'
__author__ = 'GUAGUA'
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
import time


class Image_AI_Objects_Detection():

    def __init__(self, models_path, logger, detect_target='image', detect_model='resnet', detect_result_save=False):
        self.detect_target = detect_target
        self.detect_model = detect_model
        self.models_path = models_path
        self.detect_result_save = detect_result_save
        self.logger = logger

        if self.detect_target == 'image':
            from imageai.Detection import ObjectDetection
            self.detector = ObjectDetection()
        elif self.detect_target == 'video':
            from imageai.Detection import VideoObjectDetection
            self.detector = VideoObjectDetection()
        else:
            print('target wrong')

        try:
            if self.detect_model == 'resnet':
                self.detector.setModelTypeAsRetinaNet()
                self.detector.setModelPath(self.models_path + 'resnet50_coco_best_v2.0.1.h5')
            elif self.detect_model == 'yolo_tiny':
                self.detector.setModelTypeAsTinyYOLOv3()
                self.detector.setModelPath(self.models_path + 'yolo-tiny.h5')
            else:
                raise TypeError('error')
        except:
            self.logger.writeLog('no model name {},please check the models name or path'.format(self.detect_model), level='error')

        self.detector.loadModel()
        self.logger.writeLog('model load successful', level='info')

    def detect_run(self, resource_path, save_path):
        self.logger.writeLog('detection program is running', level='info')
        if self.detect_target == 'image':
            self.image_detect_run(resource_path, save_path)
        elif self.detect_target == 'video':
            self.video_detect_run(resource_path, save_path)
        self.logger.writeLog('detection end,program quit', level='info')

    def image_detect_run(self, image_path, save_path):
        # 将检测后的结果保存为新图片
        start = time.time()
        detections = self.detector.detectObjectsFromImage(input_image=image_path,
                                                          output_image_path=save_path,
                                                          extract_detected_objects=self.detect_result_save)
        # 结束计时
        end = time.time()
        if self.detect_result_save:
            detections_result = detections[0]
        else:
            detections_result = detections
        for eachObject in detections_result:
            print(eachObject["name"] + " : " + eachObject["percentage_probability"])
            print("--------------------------------")
        print("\ncost time:", end - start)

    def video_detect_run(self, video_path, save_path):
        # 将检测后的结果保存为新图片

        start = time.time()
        detections = self.detector.detectObjectsFromVideo(input_file_path=video_path,
                                                          output_file_path=save_path,
                                                          frames_per_second=20,
                                                          log_progress=True)
        print(detections)
        # 结束计时
        end = time.time()
        print("\ncost time:", end - start)


if __name__ == '__main__':
    models_path = '..\\..\\..\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
    image_path = '..\\..\\..\\MODEL_SOFTWARE\\resources\\image_test\\'
    video_path = '..\\..\\..\\MODEL_SOFTWARE\\resources\\video_test\\'
    root_path = os.getcwd()
    detection_target = 'video'
    object_test = Image_AI_Objects_Detection(models_path, detect_target='video')
    # object_test.detect_run(resource_path=image_path + "test1.png", save_path=image_path + "test1_new.png")
    object_test.detect_run(resource_path=video_path+"traffic.mp4", save_path=video_path+"new_traffic.mp4")
