---
title: 降维-PCA
tags: 机器学习理论推导
categories: 学习
abbrlink: 62495
date: 2020-01-01 19:36:25
---
# 定义
$$
\begin{aligned}
&\lbrace\left(x_{i}, y_{i}\right)\rbrace_{i=1}^{N}\\\\
&x_{i} \in \mathbb{R}^{p}, \quad i = 1, 2, \ldots ,N
\end{aligned}
$$

Mean: $\bar{x}=\frac{1}{N} \sum_{i=1}^{N} x_{i}=\frac{1}{N} X^{\top} 1_{n}$
Covariance: $s=\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\top}=\frac{1}{N} X^{\top} H X$

If $1_N=\left(\begin{array}{l}{1} \\\\ {\vdots} \\\\ {1}\end{array}\right)_{N \times 1} \quad H_N=1_N-\frac{1}{N} 1_{N} 1_{N}^{\top} \quad, \quad \bar{x} \in \mathbb{R}^{p}, \quad s \in \mathbb{R}^{p \times p}$

# 最大投影方差
\begin{aligned}
J &=\frac{1}{N} \sum_{i=1}^{N}\left(\left(x_{i}-\bar{x}\right)^{T} u_{1}\right)^{2} \quad \text { s.t. } u_{1}^{T} u_1=1 \\\\
&=\sum_{i=1}^{N} \frac{1}{N} u_{1}^{T}\left(x_{i}-\bar{x}\right) \cdot\left(x_{i}-\bar{x}\right)^{T} u_{1} \\\\
&=u_{1}^{T} \cdot s \cdot u_{1}
\end{aligned}

\begin{cases} u_1 = \arg\max_{u_1} u_1^T\cdot s\cdot u_1^T \\\\
{\text { s.t. } u_{1}^{\top} u_1=1}
\end{cases}

$$\mathbb{L} \left(u_{1}, \lambda_{1}\right)=u_{1}^{\top} s u_{1}+\lambda\left(1-u_{1}^{T} u\right)$$
$$\frac{\partial \mathbb{L}}{\partial u_{1}}=2 s \cdot u_{1}-\lambda \cdot 2 u_{1}=0$$
$$ s u_1=\lambda u_1 $$

# 最小重构代价
数据点$x_i$在$u_1, u_2,\ldots, u_p$构成的空间投影为：
$$x_{i}=\sum_{k=1}^{p}\left(x_{i}^{\top} u_{k}\right) \cdot u_{k} $$

前$q$个特征$u_1, u_2,\ldots, u_q$构成的空间重构为：

$$\hat{x}_i = \sum^q _{i=1} (x_i^T u_k) u_k$$  


$$J=\frac{1}{N} \sum^N_{i=1} \left|\left|x_{i}-\hat{x}_i \right|\right|^2$$

\begin{aligned}
&=\frac{1}{N} \sum^N_{i=1} \left|\left|\sum^q_{q+1}\left(x_{i}^{T} u_{k}\right) u_{k}\right|\right|^{2} \\\\
& \triangleq \frac{1}{N} \sum_{i=1}^{N}  \sum_{k={q+1}}^p \left( \left(x_{i}-\bar{x}\right)^T u_{k}\right) \\\\
& =\sum_{i=1}^{N} \frac{1}{N} \sum_{k={q+1}}^p \left(\left(x_{i}-\bar{x}\right)^{T} u_{k}\right)^{2} \\\\
& =\frac{1}{N} \sum_{i=1}^{N} \sum_{q+1}^{q}\left(x_{i}^{T} u_{k}\right)^{2} \\\\
&=\sum_{k=q+1}^{p} u_{k}^T \cdot s \cdot u_{k} \\\\
& \text{s.t.} \quad u_{k}^T u_{k}=1
\end{aligned}

# SVD
 对协方差矩阵$s$进行特征分解。
$$s=G K G^{T} \quad G^{T} G=1, \quad K=\left[
\begin{matrix}
k_0 & & \\\\
& k_1 &\\\\
& & \ddots \\\\
&& & k_p \\\\
\end{matrix}
\right] , \quad
k_0 \geq k_1\geq \cdots \geq k_p 
$$

对中心化后的数据进行SVD分解：
$$HX = U\Sigma V^T $$
其中：$U^T U = I, V^T V = I, \Sigma \text{对角} $

$$s_{p\times p}=X^T H X=X^{\top} H^{\top}H X=V \Sigma {U}^{\top} \cdot {U} \Sigma V^{\top}= V \Sigma^2 V^{\top}$$
$$T_{N \times N}=H X X^{T} H=U \Sigma V^{\top} \cdot V \Sigma U^{T}=U \Sigma^{2} U^{T}$$

$T$ 和$s$有相同的特征值。
