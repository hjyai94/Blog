---
title: EM算法实现
date: 2018-09-05 10:42:01
tags: deep bayes
categories: 学习
---
# EM算法
之前在看概率图模型的时候，写过过于EM算法的内容，不过已经忘记差不多了，最近在看[1]中的材料，感觉有了新的理解，特将这些内容整理成这篇博客。
EM算法适用于存在隐变量的情况，或者说是假设存在因变量对系统进行推导。
\begin{equation}\begin{split} log\ p(X\mid \theta) &= \int q(Z)log\ p(X\mid \theta)dZ \\\\
&= \int q(Z)log \frac{p(X, Z\mid \theta)}{p(Z\mid X, \theta)}dZ \\\\
&= \int q(Z)log \frac{p(X, Z\mid \theta)}{q(Z)}dZ + \int q(Z) log \frac{q(Z)}{p(Z\mid X, \theta)}dZ\\\\
&= L(q, \theta) + KL(q\mid\mid p) \geqslant L(q, \theta)\\\\
\end{split}\end{equation}
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/EM%E7%AE%97%E6%B3%95.png)
E步是为了获得隐变量的最优值，M步是对参数求最大似然。

# [问题描述](https://cw.fel.cvut.cz/old/courses/ae4b33rpz/labs/12_em/start)
有$K$张受噪声污染的图片，每张图片大小为$H\times W$，每张图片上都有$H\times w$大小的人脸，每张人脸的位置不固定，但是高度相同，都与图片的高度一样。如下图所示，每张图片都是受严重受噪声污染的，可以看成是高斯噪声。
![]()
下图是图片的结构，其中$d_k$的位置是不固定的，$F$是不含噪声的人脸图片，$B$是不含噪声的背景图片。
![]()
下面基于EM算法来思考该问题：
可观数据：$K$张受污染的图片，$X = \lbrace X_1, ..., X_K \rbrace$
隐变量：$F$的位置，$d = \lbrace d_1, ..., d_K \rbrace$
参数：$\theta = \lbrace B, F, s^2\rbrace$
似然函数：
$$ p(X_k\mid d_k, \theta) = \prod_{ij}\begin{cases} N(X_k[i, j]\mid F[i,j-d_k], s^2), & \text {if $[i, j]\in faceArea(d_k)$} \\\\ N(X_k[i, j]\mid B[i, j], s^2), & \text{otherwise} \end{cases} $$
因为每张图片中$F$和$B$部分减去对应的$F$和$B$，这样就是纯粹的噪声，我们将噪声看做是高斯分布，所以上式就是上面的形式。
以及先验：
$p(d_k\mid a) = a[d_k] $， $\sum_j a[j] = 1$,，$a\in R^{W-w+1}$
概率模型可以写成：
$$ p(X, d\mid \theta, a) = \prod_k p(X_k\mid d_k, \theta)p(d_k\mid a) $$
按照第一张图中的EM算法步骤，进行EM算法的推导：
E步：确定隐变量的最优值
\begin{equation}\begin{split} q(d) = p(d\mid X, \theta, a) &= \prod_k p(d_k\mid x_k, \theta, a)\\\\
&= \prod_k\frac{p(X_k, d_k\mid \theta, a)}{\sum_{d_k'} p(X_k, d_k'\mid \theta, a)} \\\\
&= \prod_k\frac{p(X_k\mid d_k, theta)p(d_k\mid a)}{\sum_{d_k'} p(X_k\mid d_k', theta)p(d_k'\mid a)} \\\\
\end{split}\end{equation}
M步：对参数求最大似然。
$$ Q(\theta, a) = E_{q(d)}log\ p(X, d\mid \theta, a) \rightarrow max_{\theta, a} $$
具体推导这里就不再详细的描述了，可以参照参考文献[1]中的推导过程，最后的推导结果如下：
$$a[j] = \frac{\sum_k q( d_k = j )}{\sum_{j'}  \sum_{k'} q( d_{k'} = j')}$$
$$F[i, m] = \frac 1 K  \sum_k \sum_{d_k} q(d_k)\, X^k[i,\, m+d_k]$$
$$B[i, j] = \frac {\sum_k \sum_{ d_k:\, (i, \,j) \,\not\in faceArea(d_k)} q(d_k)\, X^k[i, j]}
	  	{\sum_k \sum_{d_k: \,(i, \,j)\, \not\in faceArea(d_k)} q(d_k)}$$
$$s^2 = \frac 1 {HWK}   \sum_k \sum_{d_k} q(d_k)
	  	\sum_{i,\, j}  (X^k[i, \,j] - Model^{d_k}[i, \,j])^2$$
 其中$Model^{d_k}[i, j]$表示由$F$和$B$组成的图片，其中$F$处于$d_k$位置。


# 程序实现
下面用EM算法来是处理对受噪声污染影响的图片，从而恢复其中的人脸图像。


# 参考文献
[1] http://deepbayes.ru/ 主要来自其中的slide
[2] https://cw.fel.cvut.cz/old/courses/ae4b33rpz/labs/12_em/start
