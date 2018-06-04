---
title: HMM和CRF
date: 2018-05-31 09:49:19
tags: 概率图模型
categories: 学习
---
# 隐马尔可夫模型
隐马尔可夫模型如下图所示：
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B.png)

## 公式表达
对于隐马尔可夫模型，通常有三组参数：
$$ trasition\ probability\ matrix\ A: p(y_t^j = 1\mid y_{t-1}^i=1)=a_{i,j} $$   $$ initial\ probability: p(y_1)\sim Multinomial(\pi_1, \pi_2, ..., \pi_M) $$    $$ emission\ probabilies: p(x_t\mid y_t^i)\sim Multinomial(b_{i,1}, b_{i,2}, ..., b_{i,K}) $$

## 推断
* 前向算法
$$ \alpha_t^k \equiv\mu_{t-1\rightarrow}(k)=P(x_1, x_2, ..., x_t, y_t^k=1) $$
$$ \alpha_t^k = p(x_t\mid y_t^k=1)\sum_i \alpha_{t-1}^ia_{i,k} $$

* 后向算法
$$ \beta_t^k \equiv \mu_{t\leftarrow t+1}(k)=P(x_{t+1}, ..., x_T\mid y_t^k=1) $$
$$ \beta_t^k = \sum_i a_{k,i}p(x_{t+1}\mid y_{t+1}^i = 1)\beta_{t+1}^i $$

对于给定观测值下的任意隐变量状态，可以同过点乘前向和后向信息得到。
$$ \gamma_t^i = p(y_t^i = 1\mid x_{1:T})\propto \alpha_t^i\beta_t^i =\sum_j \xi_t^{i,j} $$
其中有定义：
\begin{equation}\begin{split} \xi_t^{i,j} &= p(y_t^i=1,y_{t-1}^j=1, x_{1:T}) \\\\
&\propto \mu_{t-1\rightarrow t}(y_t^i=1)\mu_{t\leftarrow t+1}(y_{t+1}^i=1)p(x_{x+1}\mid y_{t+1})p(y_{t+1}\mid y_t) \\\\
&= \alpha_t^i\beta_{t+1}^j a_{i,j} p(x_{t+1}\mid y_{t+1}^i=1) \\\\
\end{split}\end{equation}
具体推导可以参考Youtube上徐亦达老师关于HMM的视频，主要思路就是message passing，运用一些迭代地技巧，可以先从最小的下标开始推导，这样比较容易发现规律，类似于数学归纳法。
在Matlab中可以将公式用向量表示，这样方便处理。
\begin{equation}\begin{split} &B_t(i)=p(x_t\mid y_t^i=1)\\\\
& A(i,j)=p(y_{t+1}^j=1\mid y_t^i=1) \\\\
& \alpha_t = (A^T\alpha_{t-1}).\ast B_t \\\\
& \beta_t = A(\beta_{t+1}.\ast B_{t+1}) \\\\
& \xi_t = (\alpha_t(\beta_{t+1}.\ast B_{t+1})^T).\ast A  \\\\
& \gamma_t = \alpha_t.\ast \beta_t \\\\
\end{split}\end{equation}

## 学习
### 监督学习
当我们知道实际状态路径时，监督学习并不是一件困难的事情(trival)，我们只需要数出转移概率和发射概率的实例就可以得到最大似然估计。
$$ a_{ij}^{ML} = \frac{\sum_n\sum_{t=2}^T y_{n,t-1}^i y_{n,t}^j}{\sum_n\sum_{t=2}^T y_{n,t-1}^i} $$
$$ b_{ik}^{ML} = \frac{\sum_n\sum_{t=2}^T y_{n,t}^i x_{n,t}^k}{\sum_n\sum_{t=2}^T y_{n,t}^i} $$
使用了伪计数的方式，可以避免零概率的出现。对于不是多项分布的情况，特别是高斯分布，我们可以利用采样的方法计算均值和方差。

### 无监督学习
当隐状态不可观的时候，可以使用Baum Welch算法进行处理完全对数似然函数，这一算法就是EM算法对HMM的求解方法。似然函数可以写成：
$$ l_c(\theta;x,y)=lop\ p(x,y)=log\prod_n (p(y_{n,1})\prod_{t=1}^T p(y_{n,t}\mid y_{n,t-1})\prod_{t=1}^T p(x_{n,t}\mid y_{n,t})) $$
完全对数似然期望是：
$$ \langle l_c(\theta;x,y)\rangle = \sum_n (\langle y_{n,1}^i \rangle_{p(y_{n,1}\mid x_n)}log\ \pi_i) + \sum_n\sum_{t=2}^T (\langle y_{n,t-1}^i y_{n,t}^j\rangle_{p(y_{n,t-1},y_{n,t}\mid x_n)}log\ a_{i,j}) + \sum_n\sum_{t=1}^T (x_{n,t}^k\langle y_{n,t}^i\rangle_{p(y_{n,t}\mid x_n)}log\ b_{i,k}) $$
* E步：
$$ \gamma_{n,t}^i = \langle  y_{n,t}^i\rangle = p(y_{n,t}^i = 1\mid x_n) $$    $$ \xi_{n,t}^{i,j} = \langle y_{n,t-1}^i y_{n,t}^j\rangle = p(y_{n,t-1}^i=1, y_{n,t}^j=1\mid x_n) $$
* M步：
$$ \pi_i=\frac{\sum_n \gamma_{n,1}^i}{N}, a_{i,j}=\frac{\sum_n\xi_{n,t}^{i,j}} {\sum_n\sum_{t=1}^{i,j} \gamma_{n,t}^i},  b_{ik}=\frac{\sum_N\sum_{t=1}^T \gamma_{n,t}^i x_{n,t}^k} {\sum_n\sum_{t=1}^{T-1}\gamma_{n,t}^i} $$

### HMM的缺点
HMM的缺点也是HMM的最要特征，就是每个观测值只与一个隐状态相关，与其他状态都无关。另外就是预测目标函数与学习目标函数不一致，HMM学习状态和观测值的联合概率$P(Y,X)$，但是我们的预测要求是需要条件概率$P(Y\mid X)$，通过这样的考虑，有了一个新的模型MEMM。

# MEMM
## 形式
















































注：本文中公式的和全部采用\sum，本文之前使用的都是\Sigma，后面也会使用\sum。
