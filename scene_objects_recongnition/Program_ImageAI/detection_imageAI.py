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

from imageai.Detection import ObjectDetection
import os
import time


class Image_AI_Objects_Detection():

    def __init__(self, model_path, detect_result_save=False):
        self.model_path = model_path
        self.detect_result_save = detect_result_save
        # 代码文件根路径
        # 创建预测类
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(self.model_path)

    def detect_run(self, image_path, save_path):
        self.detector.loadModel()
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


if __name__ == '__main__':
    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\'
    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\'
    root_path = os.getcwd()
    object_test = Image_AI_Objects_Detection(models_path + "resnet50_coco_best_v2.0.1.h5")
    object_test.detect_run(image_path=image_path + "test1.png",
                           save_path=image_path + "test1_new.png")

