---
title: EM算法
date: 2018-05-25 19:45:46
tags: 概率图模型
categories: 学习
---
# 混合高斯模型
$$ p(x_n\mid , \Sigma) = \Sigma_k \pi_k N(x\mid \mu, \Sigma_k) $$
其中$\pi_k$是混合参数，$N(x\mid \mu_k, \Sigma_k)$是其对应的高斯分布。
对于完全可观的独立同分布，对数似然可以分解为和的形式。
$$ l_c(\theta;D) = log p(x,z\mid \theta) = log p(z\mid \theta_z) + log p(z\mid z, \theta_x) $$
因为隐变量的存在，所有的变量会通过边缘概率耦合在一起。
因为对数里面有和的形式，解决有一定的困难，这样促使我们想到EM算法。

# K-Means
给定数据集$(x_1, x_2, ..., x_n)$，每个观测量都是d维的向量。k-means的目的是将n个观测量分成k个集合，在给定$z= \lbrace z_1, z_2, ..., z_n \rbrace $。为了最小化组内平方和，我们随机的初始化类别向量，然后交替进行两步，知道收敛。
* E步：将每个观测分配到聚类中，是组内平方和最小。直观上来看就是讲数据点分配到最近的中心。
$$ z_i^{(t)} = argmin_k (x_i - \mu_k^{(t)})^T\Sigma_k^{-1^{(t)}}(x_i - u_k^{(t)}) $$
* M步：重新计算中心值。
$$ \mu_k^{(t+1)} = \frac{\Sigma_i \delta(z_i^{(t)}, k)x_i} {\Sigma_i \delta(z_i^{(t)}, k)} $$
可以这么来理解EM，每个聚类可以看做具有相同的分布，比如$p(x_i\mid z_i = k)~N(x_i\mid \mu_k, \Sigma_kl) $我们希望可以学习到每个分布的参数$\mu_k$和$\Sigma$。

# EM 算法
EM算法可以有效地迭代计算存在隐变量的最大似然估计。在最大似然估计值中，我们希望能够估计出对于每个观测数据最有可能的参数。
期望完全对数似然函数：
$$ \langle l_c(\theta;x,z)\rangle = \Sigma_n\langle logp(z_n\mid \pi)\rangle_{p(z\mid x)} + \frac{1}{2}\Sigma_n\Sigma_k\langle z_n^k\rangle((x_n - \mu_k)^T\Sigma_K^{-1}(x_n-\mu_k)+log|\sigma_k|+C) $$

EM算法是利用迭代地方式最大化$\langle l_c(\theta;x,z)\rangle$。在E步中，我们利用当前参数估计量计算隐变量的充分估计量。
$$ \tau_n^{k(t)} = \langle z_n^k\rangle_{q(t)} = p(zn^k = 1\mid x,\mu^{(t)},\Sigma^{(t)}) = \frac{\pi_k^{(t)}N(x_n\mid \mu_k^{(t)},\Sigma_k^{(t)})} {\Sigma_i \pi_i^{(t)}N(x_n\mid \mu_k^{(t)},\Sigma_k^{(t)})}$$
在M步中，使用期望值来计算参数期望的最大值。
\begin{equation}\begin{split} \pi_k &= \Sigma_n \langle z_n^k \rangle_{q^{(t)}}/N = \Sigma_n \tau_n^{k(t)}/N = \langle n_k \rangle /N\\\\
\mu_k^{(t+1)} &= \frac{\Sigma_n \tau_n^{k(t)}x_n}{\Sigma_n \tau_n^{k(t)}}\\\\
\Sigma_k^{(t+1)} &= \frac{\Sigma_n \tau_n^{k(t)}(x_n-\mu_k^{(t+1)})(x_n-\mu_k^{(t+1)})^T}{\Sigma_n \tau_n^{(k(t))}}\\\\
\end{split}\end{equation}

# 比较K-means和EM
EM算法类似于K-means处理混合高斯模型，对于K-means，在E步中，我们制定每个聚类点，在M中我们假定每个点属于一个聚类重新计算聚类点。在EM算法中，我们使用概率的方式指定点为聚类点，在M步中，我们假定每个点数一个聚类以概率的重新计算聚类中心。

# EM算法理论依据
X记作观测变量，Z记作隐变量集，分布模型为$p(x,z\mid \theta)$。
如果Z是可观的我们定义对数似然函数为：$ l_c(\theta;x,z) = log\ p(x,z\mid \theta) $。对于Z是可观的，我们最大化完全对数似然。
然而，当Z不可观时，我们必须最大化边际似然，也就是不完全对数似然函数。
$$ l_c(\theta;x) = log\ p(x\mid \theta) = log\Sigma_z p(x,z\theta) $$
我们必须将不完全对数似然解耦，因为对数里面具有和的形式。
为了解决这个问题，我们引入了平均分布$q(z\mid x)$来取代z的随机性。期望完全对数似然可以定义为：
$$ \langle l_c(\theta;x,z)\rangle_q = \Sigma_z q(z\mid x,\theta)log\ p(x,z\mid \theta) $$
根据杰西不等式：
\begin{equation}\begin{split} l(\theta;x) &= log\ p(x\theta) \\\\
&= log \Sigma_z p(x,z\mid \theta) \\\\
&= log \Sigma_z q(z\mid x)\frac{p(x,z\mid \theta)}{q(z\mid x)} \\\\
&\geqslant \Sigma_z q(z\mid x)log \frac{p(x,z\mid \theta)}{q(z\mid x)} \\\\
\end{split}\end{equation}
$$ l(\theta;x) \geqslant \langle l_c(\theta;x,z)\rangle_q + H_q $$
固定数据x，定义一个函数叫做自由能：
$$ F(q,\theta) = \Sigma_z q(z\mid x) log\frac{p(x,z\mid \theta)}{q(z\mid x)} \leq l(\theta;x)$$
这样EM算法等同于在F上进行坐标上升法：
* E步：$q^{t+1} = argmax_q F(q,\theta^t) $
* M步：$\theta^{t+1} = argmax_{\theta} F(q^{t+1},\theta^t)$
$q^{t+1}(z\mid x)$是隐变量在给定数据和参数下的后验分布。$q^{t+1}=argmax_q F(q,\theta^t)=p(z\mid x,\theta^t) $
证明：这样的设置可以保证$l(\theta;x)\geqslant F(q,\theta)$
\begin{equation}\begin{split} F(p(z\mid x,\theta^t), \theta^t) &= \Sigma_z q(z\mid x) log\frac{p(x,z\mid \theta)}{q(z\mid x)}\\\\
&= \Sigma_z q(z\mid x) log\ p(x\mid \theta^t) \\\\
&= log\ p(x\mid \theta^t) \\\\
&= l(\theta^t;x) \\\\
\end{split}\end{equation}
同样可以用变分微分来表示：
$$ l(\theta;x) - F(q,\theta) = KL(q||p(z\mid x, \theta)) $$
在不失一般性的情况下，我们可以将$p(x,z\mid \theta)$定义为广义指数族分布：
$$ p(x,z\mid \theta) = \frac{1}{Z(\theta)} h(x,z) exp \lbrace \Sigma_i \theta_i f_i(x,z) \rbrace $$
如果$p(X\mid Z)$是广义线性模型，那么$f_i(x,z)=\eta_i^T(z)\xi_i(x) $。
在$q^{t+1}=p(z\mid x,\theta^t)$下，期望完全对数似然为：
\begin{equation}\begin{split} \langle l_c(\theta;x,z)\rangle_{q^{t+1}} &= \Sigma_z q(z\mid x,\theta^t)log\ p(x,z\mid \theta^t) - A(\theta) \\\\
&= \Sigma_i \theta_i^t \langle f_i(x,z)\rangle_{q(z\mid x, \theta^t)} - A(\theta) \\\\
&= \Sigma_i \theta_i^t \langle \eta_i(z)\rangle_{q(z\mid x, \theta^t)}\eta_i(x) - A(\theta) \\\\
\end{split}\end{equation}
下面分析EM算法的M步，M步可以看做是最大化期望对数似然：
\begin{equation}\begin{split} F(q,\theta) &= \Sigma_z q(z\mid x) log\frac{p(x,z\mid \theta)}{q(z\mid x)} \\\\
&= \Sigma_z q(z\mid x)log\ p(x,z\mid \theta) - \Sigma_z q(z\mid x)log\ q(z\mid x) \\\\
&= \langle l_c(\theta;x, z) \rangle_q + H_q
\end{split}\end{equation}
这样将自由能分解成两个部分，第一部分是期望完全对数似然，第二部分是熵，并且与变量$\theta$无关，这样最大自由能就等价于最大化期望完全对数似然。
$$ \theta^{t+1} = argmax_{\theta} \langle l_c(\theta;x,z)\rangle_{q^{t+1}} = argmax_{\theta} \Sigma_z q(z\mid x)log\ p(x,z\mid \theta) $$
在最优的$q^{t+1}$的情况下，这样就等同于解决标准的完全可观模型$p(x,z\mid \theta)$的最大似然问题，用$p(z\mid x, \theta)$取代包含z的充分统计量。

# 总结
EM算法是一种对隐变量模型最大似然函数的一种方法，将比较难以解决的问题分解为两步：
1. 基于当前参数和可观数据对隐变量进行估计。
2. 基于观测数据和隐变量对参数做极大似然估计。

* EM算法好的方面
 * 没有学习率参数
 * 自动限制参数
 * 低维速度快
 * 每代都可以确保调高似然
* 不好的方面
 * 会陷入局部极优
 * 比共轭梯度慢，特别是接近收敛时
 * 需要代价高的推测过程
 * 是一种最大似然或者最大后验的方法
