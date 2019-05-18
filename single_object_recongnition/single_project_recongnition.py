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
import os
import re
import webbrowser
import tensorflow as tf


class objects_recongnition():

    def __init__(self, require_method, program_method, logger):
        self.require_method = require_method
        self.program_method = program_method
        self.logger = logger
        self.models_path = '..\\..\\MODEL_SOFTWARE\\resources\\models\\imageai\\'
        self.image_path = '..\\..\\MODEL_SOFTWARE\\resources\\image_test\\'

    @staticmethod   # slim\preprocessing\inception_preprocessing
    def preprocess_for_eval(image, height, width,
                            central_fraction=0.875, scope=None):
        """Prepare one image for evaluation.

        If height and width are specified it would output an image with that size by
        applying resize_bilinear.

        If central_fraction is specified it would crop the central fraction of the
        input image.

        Args:
          image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
            [0, 1], otherwise it would converted to tf.float32 assuming that the range
            is [0, MAX], where MAX is largest positive representable number for
            int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
          height: integer
          width: integer
          central_fraction: Optional Float, fraction of the image to crop.
          scope: Optional scope for name_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
        with tf.name_scope(scope, 'eval_image', [image, height, width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            if central_fraction:
                image = tf.image.central_crop(image, central_fraction=central_fraction)

            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [height, width],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image

    def run(self, model_name=None):
        if self.program_method == 'Program_ImageAI':
            self.Program_ImageAI_run()
        elif self.program_method == 'Program_ImageNet':
            self.Program_ImageNet_run(model_name)

    def Program_ImageAI_run(self):
        from Program_ImageAI.prediction_imageAI import Image_AI_Objects_Prediction
        object_test = Image_AI_Objects_Prediction(self.models_path, self.logger, predict_model='DenseNet121')
        object_test.predict_run(image_path=self.image_path + "test5.png")

    def Program_ImageNet_run(self, model_name):
        self.model_name = model_name
        # 原始图片数据转换成tf_recode格式
        # self.Program_ImageNet_data_convert()
        # 开始训练
        # self.Program_ImageNet_data_train()
        # 数据集验证
        # self.Program_ImageNet_data_validation()
        # tensorboard 训练可视化
        self.Program_ImageNet_data_train_tensorboard()
        # 训练结果模型导出
        # self.Program_ImageNet_data_export_model()
        # 模型重载,权重固化
        # self.Program_ImageNet_data_reload()
        # 运行预测图片
        # self.Program_ImageNet_data_classify()

    def Program_ImageNet_data_convert(self):
        from Program_ImageNet.data_convert import parse_args_image
        data_dir = '..//resources//data_prepare//pic'
        params = parse_args_image(data_dir)
        params.convert_run()

    def Program_ImageNet_data_train(self):
        self.logger.writeLog('training start ', level='info')
        os.system('cd ../common/slim & \
                   python train_image_classifier.py \
                   --train_dir=../../resources/satellite/data/train_dir \
                   --dataset_name=satellite \
                   --dataset_split_name=train \
                   --dataset_dir=../../resources/data_prepare/pic \
                   --model_name={} \
                   --checkpoint_path=../../resources/models/imagenet/inception_v3.ckpt \
                   --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
                   --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
                   --max_number_of_steps=10000 \
                   --batch_size=8 \
                   --learning_rate=0.001 \
                   --learning_rate_decay_type=fixed \
                   --save_interval_secs=300 \
                   --save_summaries_secs=2 \
                   --log_every_n_steps=10 \
                   --optimizer=rmsprop \
                   --weight_decay=0.00004'.format(self.model_name))
        self.logger.writeLog('the training program has finished, please check the training results ', level='info')

    def Program_ImageNet_data_validation(self):
        self.logger.writeLog('validation start ', level='info')
        os.system('cd ../common/slim & \
                   python eval_image_classifier.py \
                   --checkpoint_path=../../resources/satellite/data/train_dir \
                   --eval_dir=../../resources/satellite/data/eval_dir \
                   --dataset_name=satellite \
                   --dataset_split_name=validation \
                   --dataset_dir=../../resources/data_prepare/pic \
                   --model_name={}'.format(self.model_name))
        self.logger.writeLog('the validation program has finished, please check the validation results ', level='info')

    def Program_ImageNet_data_train_tensorboard(self):
        os.system('tensorboard --logdir ./../resources/satellite/data/train_dir')
        webbrowser.open_new_tab('http://127.0.0.1:6006')   # 处于阻塞状态

    def Program_ImageNet_data_export_model(self):
        self.logger.writeLog('start to export model', level='info')
        os.system('cd ../common/slim & \
                   python export_inference_graph.py \
                   --alsologtostderr \
                   --model_name={} \
                   --output_file=../../resources/satellite/inception_v3_inf_graph.pb \
                   --dataset_name=satellite'.format(self.model_name))
        self.logger.writeLog('export model has finished, please check the export results ', level='info')

    def Program_ImageNet_data_reload(self):
        max_num = 0
        for root, dirs, files in os.walk('..\\resources\\satellite\\data\\train_dir'):
            for file in files:
                file_name = re.findall(r"[0-9]+.meta", file)
                if file_name:
                    file_name_num = re.findall(r'\d+', file_name[0])
                    if int(file_name_num[0]) > int(max_num):
                        max_num = file_name_num[0]
        if max_num != 0:
            self.logger.writeLog('start to reload model', level='info')
            os.system('cd ../common & \
                       python freeze_graph.py \
                       --input_graph ../resources/satellite/inception_v3_inf_graph.pb\
                       --input_checkpoint ../resources/satellite/data/train_dir/model.ckpt-{} \
                       --input_binary true \
                       --output_node_names InceptionV3/Predictions/Reshape_1 \
                       --output_graph ../resources/satellite/frozen_graph.pb'.format(max_num))
            self.logger.writeLog('reload model has finished, please check the reload results ', level='info')
        else:
            self.logger.writeLog('something is wrong,please check the model dir', level='error')

    def image_preprocessing(self, image_file):
        if self.model_name == 'inception_v3':
            with tf.Graph().as_default():
                image_data = tf.gfile.FastGFile(image_file, 'rb').read()
                image_data = tf.image.decode_png(image_data)
                image_data = self.preprocess_for_eval(image_data, 299, 299)
                image_data = tf.expand_dims(image_data, 0)
                with tf.Session() as sess:
                    image_data = sess.run(image_data)
        return image_data


    def Program_ImageNet_data_classify(self):
        model_path = '..\\resources\\satellite\\frozen_graph.pb'
        label_path = '..\\resources\\data_prepare\\pic\\label.txt'
        image_file = '..\\resources\\image_test\\test3.png'

        image_data = self.image_preprocessing(image_file)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            with tf.Session() as sess:
                softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
                predictions = sess.run(softmax_tensor, {'input:0':image_data})
                top_k = predictions.argsort()[-5:][::-1]
                for node_id in top_k:
                    score = predictions[node_id]
                    print('%s (score=%.5f)' % (str(node_id, score)))



if __name__ == '__main__':

    logger = LogHelper('single_recongnition_log')

    require_method = 'single_object_recongnition'
    program_method = 'Program_ImageNet'

    object_recongnition = objects_recongnition(require_method, program_method, logger)
    object_recongnition.run('inception_v3')
