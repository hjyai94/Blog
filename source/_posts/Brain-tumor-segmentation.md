---
title: 基于卷积神经网络的脑肿瘤分割
tags: 脑肿瘤分割
catagories: 学习
abbrlink: 59348
date: 2019-02-28 22:09:38
---
# 前言
本人一直研究脑肿瘤分割，脑肿瘤分割对于患者的后续治疗以及对疾病的检测有着重要的意义，同时也是人工智能处理医学图像的重要方向之一。目前开源的代码都比较复杂，不适合入门研究，另外 Pytorch 作为一个容易上手的深度学习框架，具有很强的灵活性，适合新手或者是科研工作者，所以本文的代码将使用深度学习框架 Pytorch1.0 和 Python3.6 进行编程构建卷积神经网络来进行脑肿瘤分割。卷积神经网络不仅在自然图像而且在医学图像在内的其他图像都有着广泛地应用。另外，卷积神经网络广泛地应用于图像的分类，检测等计算机视觉任务中。

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

## 可视化
前面我们知道 BraTs 2015 共有4个模态的数据，下面我们介绍两个能够在程序中读取医学图像的包：SimpleITK 和 Nibabel。SimpleITK能够读取的格式更加多，具体可以参考 SimpleITK 的文档。

### SimpleITK
SimpleITK 读取医学图像示例代码：
```
import SimpleITK as sitk
image = sitk.ReadImage('image.nii')
image = sitk.GetArrayFromImage(image)
```
SimpleITK 读取的图片维度是通道优先的，所以图像的维度的第1位是医学图像的维数。另外我还使用到了 SimpleITK 将输出预测结果保存为医学图像格式，这部分代码如下：
```
image = sitk.GetImageFromArray(image)
sitk.WriteImage(image, 'image.mha')
```

### Nibabel
Nibabel 读取医学图像示例代码：
```
import nibabel as nib 
image = nib.load('image.nii.gz').get_fdata()
image = image.transpse(1, 0, 2)
```
不同于 SimpleITK， Nibabel 读取图像的维度的通道数位于最后 1 位，但是 Nibabel 将图像旋转了 $90^{\circ}$，可以使用上面代码的第三行旋转为一般维数分布方式。比如脑图中，如果不旋转变换转变 Nibabel 读取的方式，脑图就是横着的。

### 3D Slicer 可视化
SimpleITK 和 Nibabel 是可以在程序中读取医学图像的包，灵活性不强，另外不适合了解医学图像的基本特性，下面我们使用 3D Slicer 来可视化我们的医学图像数据。
{% gp 5-4 %}
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/FLAIR_axis.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/FLAIR_sagittal.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/FLAIR_coronal.png)

![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/T1_axis.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/t1_sagittal.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/T1_coronal.png)
{% endgp %}
上图中，第一行为 FLAIR 模态，从左到有依次为横断面(Axis plane)，矢状面(Sagittal plane)，冠状面(Coronal plane)，第二行为 T1 模态，从左到右顺序与 FLAIR 相同，另外两个模态因为篇幅的关系不做具体地展示。下面是脑图的 mask 和手工分割肿瘤的真值(可以看做是金标准，但是个人认为还是有区别的)：
{% gp 5-4 %}
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/mask_axis.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/mask_sagittal.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/mask_coronal.png)

![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/ground_truth_axis.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/ground_truth_sagittal.png)
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/ground_truth_coronal.png)
{% endgp %}
本人使用的数据库来自于项目[2] 中，具体如何得到的 mask 数据并不是很清楚，不过通过 3D slicer 可以手工生成mask数据，这里时间有限，不准备写了。上图中第二行是三个维度的真值图，其中外围浅蓝色为剔除肿瘤之外的区域(label 0)，绿色为 edema(label 2)，红色为 enhance tumor(label 4)，深蓝色区域为 necrosis(label 1)，黄色区域为 non-enhance tumor(label 3)。注：这里使用的 Brats 2015 采用了这5个标签，之后的该数据集进行了调整，将non-enhncing 和 necrosis 合并为 label 1。

# 模型
本文使用卷积神经网络，主要结构参考[1]中的结构，不同之处在于为了方便理解，我们只是用了一条通道。模型如下图所示：
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/brain_tumor_segmentation_CNN/brain_tumor_segmentation_model.png)
训练使用交叉熵作为损失函数，利用 RMSprop 作为优化器，学习率设为 $1e-3$，可以对学习率随着epoch进行调整，这里没有改变，读者可以根据自己的想法进行调整。具体代码可以在我的github[3]中查看，喜欢记得点个小星星。

# 训练与测试
训练部分采用随机裁剪图片大小为 $75\times75\times75$ ，输入到神经网络中，最后得到$47\times47\times47$ (因为卷积中没有采用padding，所以出现了输出小于输入的情况)。测试部分使用完全大小的图像 $240 \times 240 \times 225$，输入到网络中得到大小为$212 \times 212 \times 197$，因为脑图的边缘是无效的信息，所以输出的大小与输入和真值维度不一致，不会有影响，只需要补全就行了。

# 结果
最后网络在三个子区域的分割指标 Dice 为：Whole score 0.6764， Enhancing tumor 0.4478, Tumor core 0.4819，这个结果并不是非常的好，读者如需更高准确度的分割结果，可以调整测试时候的输入图像的维数，将其保持为与训练时输入的图像维数一致，另外可以参考最新的脑肿瘤分割的工作，对代码进行改进。

# 参考文献
[1] Kamnitsas K, Ledig C, Newcombe V F J, et al. Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation[J]. Medical image analysis, 2017, 36: 61-78.
[2] https://github.com/yaq007/Autofocus-Layer
[3] https://github.com/hjyai94/Half_Pathway_DeepMedic
