---
title: 汇报list
tags: 汇报
categories: 工作
abbrlink: 32425
date: 2018-09-26 16:43:47
---
# 2018.9.10
暑假期间学习和看的论文：
学习内容：
1. 计算视觉CS131，完成了所有的homework的代码，然后传到了我的github上了。https://github.com/hjyai94/CS131_homework
2. 计算机视觉CS231看了一点，还没有看完。
3. 最近在看的深度学习和贝叶斯方法结合的一门课(http://deepbayes.ru/)，实现了一个EM算法去除噪声的代码。

看的论文：
[1] Deep Learning Markov Random Field for Semantic Segmentation
[2] Single Image Haze Removal Method Using Conditional Random Fields
[3] DehazeNet: An End-to-End System for Single Image Haze Removal

# 2018.9.27
最近看多模态的课，然后那个课上有一些论文作为阅读材料，最近主要看这些论文。

[1] Zeiler M D, Fergus R. Visualizing and understanding convolutional networks[C]//European conference on computer vision. Springer, Cham, 2014: 818-833.
1. 卷积神经网络确实可以学到层级的特征。
2. 卷积核和步长越大，网络中保存的信息就越少。

[2] Frome A, Corrado G S, Shlens J, et al. Devise: A deep visual-semantic embedding model[C]//Advances in neural information processing systems. 2013: 2121-2129.
这篇文章就是用图像和文本来做分类的问题，就是利用一个视觉模型加上一个文本模型，对于超出训练集的样本也能给出比较好的分类。

[3] Kulkarni T D, Whitney W F, Kohli P, et al. Deep convolutional inverse graphics network[C]//Advances in neural information processing systems. 2015: 2539-2547.
有两个层Encoder和Decoder,在两层之间增加了一个Graphics code，用来表示图片的变化，比如姿态、角度。

[4] Andrew G, Arora R, Bilmes J, et al. Deep canonical correlation analysis[C]//International Conference on Machine Learning. 2013: 1247-1255.
将两个神经网络看成两个非线性函数，分别来处理两个模态的数据，最大化网络输出的典型相关性。

# 2018.10.11
多模态医学图像分析
论文：
[1] Baltruaitis T, Ahuja C, Morency L P. Multimodal machine learning: A survey and taxonomy[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.
这是一篇综述多模态机器学习的综述文章，我画了这篇文章的思维导图。
 ![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/Multimodal%20Machine%20Learning.gif)

[2] Cheng X, Zhang L, Zheng Y. Deep similarity learning for multimodal medical images[J]. Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, 2018, 6(3): 248-252.
用了一个所谓的Stacked Denosing Autoencoder 来预训练DNN，用了一个5层的全连接神经网络，将不同模态在第四层输出的差作为相似性矩阵。

[3] Zhang Z, Yang L, Zheng Y. Translating and segmenting multimodal medical volumes with cycle-and shape-consistency generative adversarial network[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 9242-9251.
利用GAN合成数据来提高分割性能，不过GAN合成的数据的分布与实际数据分布是接近的。
