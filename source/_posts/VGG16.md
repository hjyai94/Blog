---
title: VGGNet
tags: Deep Learning
categories: 学习
abbrlink: 60903
date: 2018-08-31 17:43:18
---
# VGGNet模型结构
VGGNet是牛津大学视觉组(Visual Geometry Group)和Google DeepMind公司研究员共同研究出的深度卷积神经网络。VGGNet使用的比较小的卷积核(3x3)以及2x2的最大池化层，通过增加层数增强非线性性能，同时相较于7x7的卷积核而言，减少了参数。
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/vgg16.jpg)
## VGG16模型处理过程
以上面图的D为例，下面简要的阐述VGGNet模型的处理过程。
1. 输入224x224x3的图片，经过64个3x3的卷积核做两次卷积+ReLU，变成224x224x64。
2. 做MaxPool，池化尺寸为2x2，步长(stride)为2。
3. 经过128个3x3的卷积核做两次卷积+ReLU，尺寸变为112x112x128。
4. MaxPool，尺寸变为56x56x128。
5. 256个3x3的卷积核做三次卷积+ReLU，尺寸变为56x56x256。
6. MaxPool，尺寸变为28x28x256。
7. 经512个3x3的卷积核做三次卷积+ReLU，尺寸变为28x28x512。
8. MaxPool，尺寸变为14x14x512。
9. 经512个3x3的卷积核做三次卷积+ReLU，尺寸变为14x14x512。
10. MaxPool，尺寸变为7x7x512。
11. 与两层1x1x4096，一层1x1x1000进行全连接+ReLU(共三层)。
12. 通过softmax输出1000个预测结果。
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/VGG16%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/VGG%E5%8F%82%E6%95%B0.png)

# 利用VGG19实现火灾分类
主要参考[1]中的代码，另外我将自己跑出的结果贴在了我的github上，具体地址为[[4]](https://github.com/hjyai94/VGG)。



# 参考文献
[1] http://www.cnblogs.com/vipyoumay/p/7884472.html
[2] https://my.oschina.net/u/876354/blog/1634322
[3] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
[4] https://github.com/hjyai94/VGG
