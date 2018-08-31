---
title: VGG16
date: 2018-08-31 17:43:18
tags: Deep Learning
categories: 学习
---
# VGG16模型结构
VGGNet是牛津大学视觉组(Visual Geometry Group)和Google DeepMind公司研究员共同研究出的深度卷积神经
网络。VGGNet使用的比较小的卷积核(3*3)以及2*2的最大池化层，通过增加层数增强非线性性能，同时相较于7*7的
卷积核而言，减少了参数。
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/vgg16.jpg)
## VGG16模型处理过程
以上面图的D为例，下面简要的阐述VGGNet模型的处理过程。
1. 输入224*224*3的图片，经过64个3*3的卷积核做两次卷积+ReLU，变成224*224*64。
2. 做MaxPool，池化尺寸为2*2，步长(stride)为2。
3. 经过128个3*3的卷积核做两次卷积+ReLU，尺寸变为112*112*128。
4. MaxPool，尺寸变为56*56*128。
5. 256个3*3的卷积核做三次卷积+ReLU，尺寸变为56*56*256。
6. MaxPool，尺寸变为28*28*256。
7. 经512个3*3的卷积核做三次卷积+ReLU，尺寸变为28*28*512。
8. MaxPool，尺寸变为14*14*512。
9. 经512个3*3的卷积核做三次卷积+ReLU，尺寸变为14*14*512。
10. MaxPool，尺寸变为7*7*512。
11. 与两层1*1*4096，一层1*1*1000进行全连接+ReLU(共三层)。
12. 通过softmax输出1000个预测结果。
