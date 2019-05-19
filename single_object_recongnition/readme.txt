单个物体图片识别(图片内容预测)
    ImageAI--Image Prediction：ImageAI提供4种不同的算法和模型来进行图像预测，并在ImageNet-1000数据集上进行了训练。4种算法包括SqueezeNet，ResNet，InceptionV3和DenseNet。
        SqueezeNet（预测速度最快 正确率中等）
        ResNet50 （预测速度快 正确率较高）
        InceptionV3（预测速度慢 正确率高）
        DenseNet121（预测速度更慢 正确率最高

    slim: tensorflow 提供的slim库，封装了对数据进行转换，训练，验证等一系列的函数
                神经网络模型包括：用于手写数字识别等简单应用的的lenet，针对cifar数据集的cifarnet
                                  Googlenet的inceptionv1234和常用的图像识别网络vgg和resnet等
                程序运行流程 :1.准备训练数据集，分类文件夹放置训练图片，同样的分类文件夹放置对应的验证图片
                              2.用data_convert.py将源图片数据文件夹整个转换成tfrecord文件
                              3.开始训练，配置训练时的参数，包括训练过程存储文件夹，数据集名称，数据集存储路径
                                模型选择，模型结构载入，选择训练模型的网络层(一般都只训练低层网络和全连接层，去掉trainable_scop
                                后即会训练整个网络),最大训练步数和训练批次(视gpu内存而定)，学习率大小和是否固定，存储训练检查点的间隔
                                optimizer：优化器，weight_decay：模型中所有的参数的二次正则化超参数
                              4.训练结束后用eval_image_classifier.py进行模型效果验证
                              5.验证结束后用export_inference_graph.py导出训练后的模型，生成一个.pb文件，该文件为模型结构
                              6.用freeze_graph.py将checkpoint里面报错的模型权重参数固化进pb文件(模型结构)中，至此一个训练后的神经网络搭建成功
                                选中output_node_names：模型的输出节点
                              7.图片预测，输入图片，图片读取+预处理(slim提供了不同的网络不同过得预处理方法)
                                读取固话后的模型(frozen_graph.pb)，开始前向传播进行预测