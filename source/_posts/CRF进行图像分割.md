---
title: CRF进行图像分割
tags: 概率图模型
categories: 学习
abbrlink: 39156
date: 2018-11-29 15:58:49
---
# 写在前面的话
最近，十分困惑条件随机场是如何工作的，为什么可以加在卷积神经网络的后面作为后处理的部分。虽然理论部分前面的博客也有写过，做过一些总结，不过因为没有实现过代码，所以仍有困惑解决不了，每念至此，心绪不宁，遂作此文，以供参考。

# 具体实现
本文将实现全连接随机场对非RGB的图像进行分割，主要参考文献[1]以及对应的[github](https://github.com/lucasb-eyer/pydensecrf)代码，另外本文需要安装pydensecrf，可以通过`pip install pydensecrf`安装，安装时需注意，pydensecrf依赖于cython，需要先安装cython。


## 对非RGB图像分割
本文的代码放在了我的github中命名为CRF的仓库库中，[链接地址](https://github.com/hjyai94/CRF/blob/master/examples/Non%20RGB%20Example.ipynb)，这里的代码来自于[pydensecrf](https://github.com/lucasb-eyer/pydensecrf)。

### 一元势
一元势包含了每个像素对应的类别，这些可以来自随机森林或者是深度神经网络的softmax。这里，我们共有两个类别，一个是前景，一个是背景，这里大小设置为$400\times 512$。我们建立了两个二维的高斯分布，并且平面显示。

```python
from scipy.stats import multivariate_normal

H, W, NLABELS = 400, 512, 2

# This creates a gaussian blob...
pos = np.stack(np.mgrid[0:H, 0:W], axis=2)
print(pos.shape)
rv = multivariate_normal([H//2, W//2], (H//4)*(W//4))
probs = rv.pdf(pos)
print(probs.shape)
# ...which we project into the range [0.4, 0.6]
probs = (probs-probs.min()) / (probs.max()-probs.min())
probs = 0.5 + 0.2 * (probs-0.5)

# The first dimension needs to be equal to the number of classes.
# Let's have one "foreground" and one "background" class.
# So replicate the gaussian blob but invert it to create the probability
# of the "background" class to be the opposite of "foreground".
probs = np.tile(probs[np.newaxis,:,:],(2,1,1))
probs[1,:,:] = 1 - probs[0,:,:]

# Let's have a look:
plt.figure(figsize=(15,5))
plt.subplot(1,2,1); plt.imshow(probs[0,:,:]); plt.title('Foreground probability'); plt.axis('off'); plt.colorbar();
plt.subplot(1,2,2); plt.imshow(probs[1,:,:]); plt.title('Background probability'); plt.axis('off'); plt.colorbar();
```
![output_9_1.png](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/CRF%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/output_9_1.png)


### 使用一元势进行推断
这里我们可以使用一元势进行推断，也就是说这里我们不考虑像素间的相互关联。这样做并不是很好的推断，但是可以这么做。
```python
# Inference without pair-wise terms
U = unary_from_softmax(probs)  # note: num classes is first dim
d = dcrf.DenseCRF2D(W, H, NLABELS)
d.setUnaryEnergy(U)

# Run inference for 10 iterations
Q_unary = d.inference(10)

# The Q is now the approximate posterior, we can get a MAP estimate using argmax.
map_soln_unary = np.argmax(Q_unary, axis=0)

# Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
map_soln_unary = map_soln_unary.reshape((H,W))
# And let's have a look.
plt.imshow(map_soln_unary); plt.axis('off'); plt.title('MAP Solution without pairwise terms');
```
![output_12_0.png](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/CRF%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/output_12_0.png)

### 二元势
图像处理中，我们经常使用像素间的双边关系，也就是说，我们认为有相似颜色的或者是相似的位置的像素认为是同一类。下面我们建立这样的双边关系。

```python
NCHAN=1

# Create simple image which will serve as bilateral.
# Note that we put the channel dimension last here,
# but we could also have it be the first dimension and
# just change the `chdim` parameter to `0` further down.
img = np.zeros((H,W,NCHAN), np.uint8)
img[H//3:2*H//3,W//4:3*W//4,:] = 1

plt.imshow(img[:,:,0]); plt.title('Bilateral image'); plt.axis('off'); plt.colorbar();

# Create the pairwise bilateral term from the above image.
# The two `s{dims,chan}` parameters are model hyper-parameters defining
# the strength of the location and image content bilaterals, respectively.
pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)

# pairwise_energy now contains as many dimensions as the DenseCRF has features,
# which in this case is 3: (x,y,channel1)
img_en = pairwise_energy.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(img_en[0]); plt.title('Pairwise bilateral [x]'); plt.axis('off'); plt.colorbar();
plt.subplot(1,3,2); plt.imshow(img_en[1]); plt.title('Pairwise bilateral [y]'); plt.axis('off'); plt.colorbar();
plt.subplot(1,3,3); plt.imshow(img_en[2]); plt.title('Pairwise bilateral [c]'); plt.axis('off'); plt.colorbar();
```
![output_17_0.png](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/CRF%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/output_17_0.png)
![output_17_0.png](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/CRF%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/output_18_0.png)

### 使用完整的条件随机场进行推断
下面我们将一元势与二元势结合起来进行推断，执行不同的迭代次数，有下面的结果。
```python
d = dcrf.DenseCRF2D(W, H, NLABELS)
d.setUnaryEnergy(U)
d.addPairwiseEnergy(pairwise_energy, compat=10)  # `compat` is the "strength" of this potential.

# This time, let's do inference in steps ourselves
# so that we can look at intermediate solutions
# as well as monitor KL-divergence, which indicates
# how well we have converged.
# PyDenseCRF also requires us to keep track of two
# temporary buffers it needs for computations.
Q, tmp1, tmp2 = d.startInference()
for _ in range(5):
    d.stepInference(Q, tmp1, tmp2)
kl1 = d.klDivergence(Q) / (H*W)
map_soln1 = np.argmax(Q, axis=0).reshape((H,W))

for _ in range(20):
    d.stepInference(Q, tmp1, tmp2)
kl2 = d.klDivergence(Q) / (H*W)
map_soln2 = np.argmax(Q, axis=0).reshape((H,W))

for _ in range(50):
    d.stepInference(Q, tmp1, tmp2)
kl3 = d.klDivergence(Q) / (H*W)
map_soln3 = np.argmax(Q, axis=0).reshape((H,W))

img_en = pairwise_energy.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(5 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(20 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(75 steps, KL={:.2f})'.format(kl3)); plt.axis('off');
```
![output_21_0.png](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/CRF%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2/output_21_0.png)


# 参考文献
[1] Krähenbühl P, Koltun V. Efficient inference in fully connected crfs with gaussian edge potentials[C]//Advances in neural information processing systems. 2011: 109-117.