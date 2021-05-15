---
title: 支持向量机-硬间隔SVM
date: 2020-01-03 21:03:11
tags: 机器学习理论推导
categories: 学习
---
# 定义
$$
\begin{aligned}
&\lbrace\left(x_{i}, y_{i}\right)\rbrace_{i=1}^{N}\\\\
&x_{i} \in \mathbb{R}^{p}, \quad y_{i} \in \lbrace -1,+1\rbrace
\end{aligned}
$$

最大间隔分类器 $\max margin(w, b)$
$ \text{s.t.}\ y_i(w^Tx_i + b)> 0, \text{for}\ \forall i=1, \ldots, N$

$$margin(w, b) = \min distance(w, b, x_i) \\\\ =\min \frac{1}{||w||}|w^Tx_i +b|$$

因为$y_i(w^Tx_i + b)> 0$，所以$\exists r>0, s.t. \min y_i(w^Tx_i + b)=r $
令$r=1$，
$$\max_{w,b} \min_{x_i} \frac{1}{||w||}(w^Tx_i +b) = \max_{w,b} \frac{1}{||w||} \min_{x_i}y_i(w^Tx_i +b) = \max_{w,b} \frac{1}{||w||}$$


\begin{cases}
\max_{w,b}\ \frac{1}{||w||}  \\\\
s.t. \min y_i(w^Tx_i + b) = 1, i=1, \ldots, N
\end{cases}

原问题：
\begin{cases}
\min_{w,b}\ \frac{1}{2} w^T w \\\\
s.t. y_i(w^Tx_i + b) \geq 1, i=1, \ldots, N
\end{cases}

# 求解
利用拉格朗日求解上面问题：
$$\mathbb{L}(w, b, \lambda)=\frac{1}{2}w^T w + \sum_{i=1}^N \lambda_i(1-y_i(w^T x_i + b)) $$
对上式求道并令其为$0$：
$$\frac{\partial \mathbb{L}}{\partial b} =0 \rightarrow \sum_{i=1}^{N} \lambda_{i} y_{i}=0$$
将上式带入$\mathbb{L}(w, b, \lambda)$

$$
\begin{aligned}
\mathbb{L}(w, b, \lambda) &=\frac{1}{2} w^{\top} w+\sum_{i=1}^{N} \lambda_{i}-\sum_{i=1}^{N} \lambda_{i} y_{i}\left(w^{\top} x_{i}+b\right) \\\\
&=\frac{1}{2} w^{\top} w+\sum_{i=1}^{n} \lambda_{i}-\sum_{i=1}^{N} \lambda_{i} y_{i} w^{\top} x_{i}+\sum_{i=1}^{N} \lambda_{i} y_{i} b \\\\
&=\frac{1}{2} w^{\top} w+\sum_{i=1}^{N} \lambda_{i}-\sum_{i=1}^{N} \lambda_{i} y_{i} w^{\top} x_{i}
\end{aligned}
$$

$$\frac{\partial \mathbb{L}}{\partial w}=w -\sum_{i=1}^{n} \lambda_{i} y_{i} x_{i} \triangleq 0 \Rightarrow {w^{*}=\sum_{i=1}^{N} \lambda_{i} y_{i} x_{i}}$$

将$w^{\star}, b^{\star}$带入 $\mathbb{L}(w, b, \lambda)$：
$$\mathbb{L}(w, b, \lambda) = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} x_{i}^{\top} x_{j}+\sum_{i=1}^{N} \lambda_{i}$$

原问题的对偶问题为：
\begin{cases}
\max -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} x_{i}^{\top} x_{j}+\sum_{i=1}^{N} \lambda_{i} \\\\
s.t. \lambda_i \geq 0
\end{cases}

# KKT条件
\begin{cases}
\frac{\partial \mathbb{L}}{\partial w}=0 \quad \frac{\partial \mathbb{L}}{\partial b}=0 \quad \frac{\partial \mathbb{L}}{\partial \lambda_i}=0  \\\\
1-y_{i}\left(w^T x_i+b\right)=0 \\\\
\lambda_i \geq 0 \\\\
1-y_{i}\left(w^{\top} x_{i}+b\right) \leqslant 0
\end{cases}

