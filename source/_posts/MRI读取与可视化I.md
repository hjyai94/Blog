---
title: MRI读取与可视化I
date: 2018-10-18 19:00:40
tags: Medical Image
categories: 学习
---
今天在网上看了一些读取MRI文件的方法，中文的博客并不是很多，另外很多并不适合我的文件格式，本文主要是针对MRI中采用NIFTI(.nii.g其中gz是压缩文件)格式的文件，并进行可视化分析。
# NIBabel
NIBabel是一个常见的读写神经医学文件的python库，包括ANALYZE, GIFTI, NIfTI2, MINC1, MINC2, MGH和ECAT，还有Philips PAR/REC。
下面我们用一张大脑的[MRI图片](http://nipy.org/nibabel/_downloads/someones_epi.nii.gz)，来说这个库的使用，以及MRI文件的格式。
```python
import nibabel as nib
img = nib.load('downloads/someones_epi.nii.gz')
img_header = img.get_header()
print('Header: ', img_header)
img_data = img.get_fdata()
print('img_data shape: ', img_data.shape)
```
一个格式为NIFTI的格式文件，通常包括头文件和相应的图像文件。图像文件时三维的，上面的MRI图片的shape为(53, 61, 33)。下面我们将三个维度的中间slice进行可视化(因为不会直接将三维的图像可视化)。

# 可视化
下面我们显示中间的slice：
```python
plt.subplot(1, 3, 1)
plt.imshow(img_data[26, :, :])
plt.subplot(1, 3, 2)
plt.imshow(img_data[:, 30, :])
plt.subplot(1, 3, 3)
plt.imshow(img_data[:, :, 15])
plt.show()
```

# 下面的工作
利用pytorch读取文件，训练神机网络。(可能不会上传到本博客中)




# 参考资料
[1] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-coordinates-in-the-image-data-array