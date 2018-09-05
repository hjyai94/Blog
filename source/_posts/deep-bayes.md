---
title: Deep|Bayes summer school笔记
date: 2018-09-05 10:42:01
tags: Bayes
categories: 学习
---
# EM算法
EM算法适用于存在隐变量的情况，或者说是假设存在因变量对系统进行推导。
\begin{equation}\begin{split} log\ p(X\mid \theta) = \int q(Z)log\ p(X\mid \theta)dZ &= \int q(Z)log \frac{p(X, Z\mid \theta)}{p(Z\mid X, \theta)}dZ \\\\
&= \int q(Z)log \frac{p(X, Z\mid \theta)}{q(Z)}dZ + \int q(Z) log \frac{q(Z)}{p(Z\mid X, \theta)}dZ\\\\
&= L(q, \theta) + KL(q\mid\mid p) \geqslant L(q, \theta)\\\\
\end{split}\end{equation}

![]()
# 参考文献
[1] http://deepbayes.ru/ 主要来自其中的slide
