# coding:utf-8
from __future__ import absolute_import
import argparse
import os
from tfrecord import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorflow-data-dir', default='pic/')
    parser.add_argument('--train-shards', default=2, type=int)
    parser.add_argument('--validation-shards', default=2, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--dataset-name', default='satellite', type=str)
    return parser.parse_args()


class parse_args_image():
        """
        command_args:需要有以下属性：
        command_args.train_directory  训练集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
        command_args.validation_directory 验证集所在的文件夹。这个文件夹下面，每个文件夹的名字代表label名称，再下面就是图片。
        command_args.labels_file 一个文件。每一行代表一个label名称。
        command_args.output_directory 一个文件夹，表示最后输出的位置。
        command_args.train_shards 将训练集分成多少份。
        command_args.validation_shards 将验证集分成多少份。
        command_args.num_threads 线程数。必须是上面两个参数的约数。
        command_args.class_label_base 很重要！真正的tfrecord中，每个class的label号从多少开始，默认为0（在models/slim中就是从0开始的）
        command_args.dataset_name 字符串，输出的时候的前缀。
        图片不可以有损坏。否则会导致线程提前退出。
        """
        def __init__(self, tensorflow_data_dir, train_shards=2, validation_shards=2, num_threads=2, dataset_name='satellite'):
            self.tensorflow_data_dir = tensorflow_data_dir
            self.train_directory = os.path.join(tensorflow_data_dir, 'train')
            self.validation_directory = os.path.join(tensorflow_data_dir, 'validation')
            self.output_directory = tensorflow_data_dir
            self.labels_file = os.path.join(tensorflow_data_dir, 'label.txt')

            self.train_shards = train_shards
            self.validation_shards = validation_shards
            self.num_threads = num_threads
            self.dataset_name = dataset_name

        def convert_run(self):
            if os.path.exists(self.labels_file) is False:
                all_entries = os.listdir(self.train_directory)
                dirnames = []
                for entry in all_entries:
                    if os.path.isdir(os.path.join(self.train_directory, entry)):
                        dirnames.append(entry)
                with open(params.labels_file, 'w') as f:
                    for dirname in dirnames:
                        f.write(dirname + '\n')

            main(self)


if __name__ == '__main__':
    data_dir = '..//..//resources//data_prepare//pic'

    params = parse_args_image(data_dir)
    params.convert_run()


