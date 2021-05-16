---
title: 支持向量机-软间隔SVM
tags: 机器学习理论推导
categories: 学习
abbrlink: 60490
date: 2020-01-04 12:27:07
---
# 定义
$$
\begin{aligned}
&\lbrace\left(x_{i}, y_{i}\right)\rbrace_{i=1}^{N}\\\\
&x_{i} \in \mathbb{R}^{p}, \quad y_{i} \in \lbrace -1,+1\rbrace
\end{aligned}
$$

$$\min \frac{1}{2} w^{\top} w+\operatorname{loss}$$
1. $$loss =\sum_{i=1}^{N} I\left\lbrace_{i}\left(w^{\top} x_{i}+b\right)<1\right\rbrace$$
2. $loss:$距离
$$
\begin{aligned}
&\text{如果}\  y_{i}\left(w^{\top} x_{i}+b\right) \geqslant 1, \quad \text {loss} = 0\\\\
&\text{如果}\ y_{i}\left(w^{\top} x_{i}+b\right)<1, \quad \text { loss }=1-y_{i}\left(w^{\top} x_{i}+b\right)
\end{aligned}
$$

$$\operatorname{loss}= \max \left\lbrace 0,1-{y_{i}\left(w^{\top} x_{i}+b\right)}\right\rbrace$$

\begin{cases}
\min_{w, b} \frac{1}{2} w^{\top} w+C \sum_{i=1}^{N} \max \left\lbrace 0,1-y_{i}(w^{\top} x_{i}+b)\right\rbrace \\\\
\text { s.t. } y_{i}\left(w^{\top} x_{i}+b\right) \geqslant 1
\end{cases}

引入$\xi_{i}=1-y_{i}\left(w^{T} x_{i}+b\right), \quad \xi_{i} \geqslant 0$
\begin{cases}
\min_{w, b} \frac{1}{2} w^{T} w+C\sum_{i=1}^{N} \xi_{i} \\\\
\text { s.t. } y_{i}\left(w^{\top} x_{i}+b\right) \geqslant 1-\xi_{i} \quad \xi_i \geq 0\\\\
\end{cases}