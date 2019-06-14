  场景里多个物体检测方案
    ImageAI--Object Detection：ImageAI中提供的物体检测功能，对图像进行检测提取多个目标
                             ：模型--RetinaNet

    OpenCV--cv2.dnn：OpenCV提供读取用caffe或TensorFlow训练好的网络模型，通过模型前向传播进行探测

    program_objectdetection: Google公司提供了TensorFlow object detection API，原理为R-CNN(第一个将深学习应用到目标检测的算法)
                             关键词：提取框(启发式搜索)，提取特征(CNN提取特征)，图像分类(svm分类)，非极大抑制
                             改进：fast R-CNN：使用ROI池化层提取特征，使用神经网络进行分类
                                   faster R-CNN：使用RPN网络取代selective search进行预选框提取
                             objectdetection使用步骤：
                                    1.安装TensorFlow Object Detection API(proto文件编译，slim文件路径)
                                    2.导入预训练模型
                                    3.用预训练模型进行测试训练和检测
                                    4.训练新的模型
                                    5.导出并测试
                             注： 代码编写暂停
