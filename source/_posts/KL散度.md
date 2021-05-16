---
title: KL散度
tags: KL散度
categories: 学习
abbrlink: 62738
date: 2018-11-27 15:59:48
---
很久没有推导过公式了，感觉水平退步显著，今日看变分推断内容，看到了计算两个高斯分布间的KL散度，下面我自己推导了一下。

# 高斯分布间的KL散度
现有先验分布$p_{\theta}(z) = \boldsymbol{N}(0, \boldsymbol{I})$，后验分布$q_{\phi}(\boldsymbol{z}\mid \boldsymbol{x}^{(i)})$同样是高斯分布。变量$z$的维数是$J$。其中，$\boldsymbol{u}$和$\boldsymbol{\sigma}$记作点$i$的均值和标准差。另外，$\mu_j$和$\sigma_j$是均值和方差向量的第$j$个因子。
KL散度的公式如下：
\begin{equation}\begin{split}
D_{KL}(q_{\phi}(\boldsymbol{z})|| p_{\theta}(\boldsymbol{z})) &= \int q_{\phi}(\boldsymbol{z}) log \frac{q_{\phi}(\boldsymbol{z})} {p_{\theta}(\boldsymbol{z})} d\boldsymbol{z}\\\\
&= \int q_{\phi}(\boldsymbol{z}) log q_{\phi}(\boldsymbol{z}) d\boldsymbol{z} - \int q_{\phi}(\boldsymbol{z}) log p_{\theta}(\boldsymbol{z}) d\boldsymbol{z} \\\\
\end{split}\end{equation}
第二项如下所示(因为先写的第二项，小声bb.jpg)：
\begin{equation}\begin{split}
\int q_{\phi}(\boldsymbol{z}) log p_{\theta}(\boldsymbol{z}) d\boldsymbol{z} &= \int \mathcal{N}(\boldsymbol{z;\mu, \sigma^2}) log \mathcal{N}(\boldsymbol{z; 0, I})d \boldsymbol{z} \\\\
&= \int -\frac{1}{2} log{2\pi}\ q_{\phi}(\boldsymbol{z}) -\frac{z^2}{2} q_{\phi}(\boldsymbol{z}) d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \int q_{\phi}(\boldsymbol{z}) \boldsymbol{z}^2 d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \int q_{\phi}(\boldsymbol{z}) (\boldsymbol{z- \mu + \mu})^2 d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \int q_{\phi}(\boldsymbol{z}) [(\boldsymbol{z- \mu})^2 + \boldsymbol{\mu}^2 +2(\boldsymbol{z - \mu})\boldsymbol{\mu}] d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \int q_{\phi}(\boldsymbol{z}) [(\boldsymbol{z- \mu})^2 + \boldsymbol{\mu}^2)] d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \int q_{\phi}(\boldsymbol{z}) (\boldsymbol{\sigma}^2 + \boldsymbol{\mu}^2) d\boldsymbol{z} \\\\
&= -\frac{J}{2}log(2\pi) -\frac{1}{2} \sum_{j=1}^{J} (\mu_j^2 + \sigma_j^2)
\end{split}\end{equation}

同样，第一项可以写成如下的形式：
\begin{equation}\begin{split}
\int q_{\phi}(\boldsymbol{z}) log q_{\phi}(\boldsymbol{z}) d\boldsymbol{z} &= -\frac{J}{2}log(2\pi) -\frac{1}{2} \sum_{j=1}^{J} (1 + log\ \sigma_j^2)
\end{split}\end{equation}
将上面两项合并一起：
\begin{equation}\begin{split}
D_{KL}(q_{\phi}(\boldsymbol{z}) || p_{\theta}(\boldsymbol{z})) &= \int q_{\phi}(\boldsymbol{z}) log (q_{\phi}(\boldsymbol{z}) - p_{\theta}(\boldsymbol{z})) d\boldsymbol{z}\\\\
&= -\frac{1}{2} \sum_{j=1}^{J} (1 + log\ \sigma_j^2) + \frac{1}{2} \sum_{j=1}^{J} (\mu_j^2 + \sigma_j^2) \\\\
&= \frac{1}{2} \sum_{j=1}^{J} (-1 - log\ \sigma_j^2 + \mu_j^2 + \sigma_j^2)
\end{split}\end{equation}

# 参考文献
Kingma D P. Variational inference & deep learning: A new synthesis[D]. 2017.
