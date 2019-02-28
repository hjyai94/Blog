---
title: 基于卷积神经网络的脑肿瘤分割
date: 2019-02-28 22:09:38
tags: 脑肿瘤分割
catagories: 学习
---
# 前言
本人一直研究脑肿瘤分割，目前开源的代码都比较复杂，不适合入门研究，另外我对 Pytorch 有独特的兴趣，所以本文的代码将使用深度学习框架 Pytorch1.0 和 Python3.6 进行编程， 另外还需要 SimpleITK 来保存分割结果，用 Nibabel 来读取数据。
脑肿瘤分割对于患者的后续治疗以及对疾病的检测有着重要的意义，同时也是人工智能处理医学图像的重要方向之一。卷积神经网络具有强大的学习能力，并且对于图像的特征具有比较好的表现效果，所以卷积神经网络不仅在自然图像而且在医学图像在内的其他图像同样有着广泛地应用。

# 数据
我们这里使用 BraTs2015 的部分数据集，数据集可以从[这里下载](https://github.com/yaq007/Autofocus-Layer)，完整的BraTs2015 数据集可以在这里[注册下载](https://www.smir.ch/BRATS/Start2015)。

本文使用的数据集共有20例样本用于训练，54例样本用于测试(可自行调整)，每个样本中共有4个模态的数据和Mask和真值数据，其中4个模态分别为FLAIR， T1，T1c，T2。真值数据共有5个标签：
* label 1: necrosis
* label 2: edema
* label 3: non-enhacing tumor
* label 4: enhancing tumor
* label 0：everything else

脑肿瘤分割主要有3个部分，Whole tumor， Tumor core， Enhance tumor。这3个部分的标签如下所示：
* Whole tumor: label 1, 2, 3, 4
* Enhance tumor: label 4
* Tumor core: 1, 3, 4

BraTs2015 使用 Dice 作为评价指标，这个评价指标主要是衡量预测结果与真值之间重叠部分。 Dice 的公式计算如下：
$$ Dice = \frac{2 TP}{2TP + FP + FN} $$

# 模型
本文使用卷积神经网络，主要结构参考[1]中的结构，不同之处在于为了方便理解，我们只是用了一条通道。模型如下图所示：

训练使用交叉上作为损失函数，利用

# 参考文献
[1] Kamnitsas K, Ledig C, Newcombe V F J, et al. Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation[J]. Medical image analysis, 2017, 36: 61-78.
