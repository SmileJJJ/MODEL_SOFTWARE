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
from imageai.Prediction import ImagePrediction
import os
import time


class Image_AI_Objects_Prediction():

    def __init__(self, models_path, predict_model='InceptionV3'):
        self.models_path = models_path
        self.predict_model = predict_model
        self.prediction = ImagePrediction()

        if self.predict_model == 'InceptionV3':
            self.prediction.setModelTypeAsInceptionV3()
            path = self.models_path + 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
            self.prediction.setModelPath(path)
        elif self.predict_model == 'SqueezeNet':
            self.prediction.setModelTypeAsSqueezeNet()
            path = self.models_path + 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5'
            self.prediction.setModelPath(path)
        elif self.predict_model == 'ResNet50':
            self.prediction.setModelTypeAsResNet()
            path = self.models_path + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
            self.prediction.setModelPath(path)
        elif self.predict_model == 'DenseNet121':
            self.prediction.setModelTypeAsDenseNet()
            path = self.models_path + 'DenseNet-BC-121-32.h5'
            self.prediction.setModelPath(path)
        else:
            print('no model name ' + self.predict_model)
        self.prediction.loadModel()

    def predict_run(self,image_path):
        start = time.time()
        # 预测图片，以及结果预测输出数目
        predictions, probabilities = self.prediction.predictImage(image_path, result_count=5)
        # 结束计时
        end = time.time()
        # 输出结果
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction, " : ", eachProbability)
        print("\ncost time:", end - start)


if __name__ == '__main__':
    models_path = 'F:\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
    image_path = 'F:\\MODEL_SOFTWARE\\resources\\image_test\\'
    root_path = os.getcwd()

    object_test = Image_AI_Objects_Prediction(models_path, predict_model='InceptionV3')
    object_test.predict_run(image_path=image_path + "test5.png")
