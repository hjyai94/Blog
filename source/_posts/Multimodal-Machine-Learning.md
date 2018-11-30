---
title: 多模态机器学习总结与多模态医学图像处理
date: 2018-10-03 17:25:45
tags: 多模态机器学习
categories: 学习
---
# 多模态机器学习
下面是我参考文章[1]总结出来多模态机器学习中存在的挑战，以及目前所使用的方法的思维导图。
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/Multimodal%20Machine%20Learning.gif)

# 多模态医学图像数据集
1. 脑肿瘤分割数据集
[1] Menze B H, Jakab A, Bauer S, et al. The multimodal brain tumor image segmentation benchmark (BRATS)[J]. IEEE transactions on medical imaging, 2015, 34(10): 1993.
https://www.med.upenn.edu/sbia/brats2018.html
这个数据集是多模态核磁成像数据集，对肿瘤进行不同尺度的扫描，然后进行多模态分割。

2. 融合数据库
http://www.med.harvard.edu/AANLIB/
http://www.metapix.de/

3. IXI数据库
http://brain-development.org/ixi-dataset/
这个数据库包含600张MRI图像，可以用于预训练，下面的文章使用了在这个方法。
Simonovsky M, Gutiérrez-Becker B, Mateus D, et al. A deep metric for multimodal registration[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2016: 10-18.

4. Radiology Objects in COntext
https://github.com/razorx89/roco-dataset
可以用于训练生成模型生成图片标题，用于分类图片的分类模型，还有基于内容的图像检索。

5. Grand Challenges in Biomedical Image Analysis
https://grand-challenge.org/
这个网站是对已经发表的论文或者是目前使用的算法提供一个竞赛的平台，里面有很多公开的医学图像数据集，其中包括1， 6， 7中的数据集。

6. Motion Tracking Challenge
http://stacom.cardiacatlas.org/motion-tracking-challenge/
这个数据库中包含有MRI和3D ultrasound 两个模态的图像。

7. ​Automatic Intervertebral Disc Localization and Segmentation from 3D Multi-modality MR (M3) Images 
https://ivdm3seg.weebly.com/data.html
这个数据集是MRI图像，用于定位和分割。

8. 一些数据集
https://sites.google.com/site/aacruzr/image-datasets

9. Image Registration Evaluation Project
http://www.insight-journal.org/rire/
用作image registratration效果评价的数据集

10. DICOM image sample sets
有一些数据供下载，部分是多模态的，数量比较少，只是一些例子，另外，看起来是注册是要收费的样子。

11. https://ida.loni.usc.edu/login.jsp
一些脑神经科学的医学图像数据库

12. ANDI(阿尔兹海默症)
http://adni.loni.usc.edu/data-samples/access-data/

13. Cancer Imaging Archive
http://www.cancerimagingarchive.net/

# 参考资料
[1] Baltrušaitis T, Ahuja C, Morency L P. Multimodal machine learning: A survey and taxonomy[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.



