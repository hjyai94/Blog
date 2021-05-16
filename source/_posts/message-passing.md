---
title: message-passing
tags: 概率图模型
categories: 学习
abbrlink: 58864
date: 2018-04-18 10:52:19
---
# 变量消除的缺点
elimination algorihthm中会有clique中重复使用的情况，message passing将重复使用的clique保留下来，这样可以减少运算复杂度。

# Elimination on a tree
将从i开始的变量消除记作$m_{ji}(x_i)$，并且是$x_i$的函数。
$$m_{ji}(x_i)=\sum_{x_j}(\psi(x_j) \psi(x_i,x_j)\prod_{k\in N(j)\j} m_{kj}(x_j))$$
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/elimination%20on%20a%20tree.png)
$m_{ji}(x_i)$能够表示从$x_j$到$x_i$的置信。

# Two-pass Algorithm
算法的实施的具体步骤，确定一个根节点，从其他节点中收集信息到这个根节点，然后回到分布的信息。直到某一节点中包含了除却继续传播节点的所有信息，这样就计算信息，然后传播到剩下的节点。


# 参考文献
[1] http://www.cs.cmu.edu/~epxing/Class/10708-14/lecture.html
[2] http://people.eecs.berkeley.edu/~jordan/prelims/?C=N;O=A
注：本文主要参考[1]中第5讲视频以及笔记，参考[2]中第4章。
