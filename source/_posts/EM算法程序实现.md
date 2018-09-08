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
有$K$张受噪声污染的图片，每张图片大小为$H\times W$，每张图片上都有$H\times w$大小的人脸，每张人脸的位置不固定，但是高度相同，都与图片的高度一样。如下图所示，每张图片都是受严重受噪声污染的，可以看成是高斯噪声。可以从这里得到[数据集](https://drive.google.com/open?id=1NLOHNhqdDBG6rWk8lOjzm3u3vDyS_9WZ)，该数据集为.mat格式，其中包含有500张$45\times 60$的图片，其中人脸大小为$45\times 36$
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/EM%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0/%E5%99%AA%E5%A3%B0%E5%9B%BE%E7%89%87.png)
下图是图片的结构，其中$d_k$的位置是不固定的，$F$是不含噪声的人脸图片，$B$是不含噪声的背景图片。
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/EM%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0/%E5%9B%BE%E7%89%87%E7%BB%93%E6%9E%84.png)
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
程序思路为
1. 实现对数似然；
2. 实现variational lower bound；
3. 实现E步；
4. 实现M步；
5. 将循环执行EM步，知道满足结束条件。
## 具体程序
* 实现对数似然，似然函数为：$$ p(X_k\mid d_k, \theta) = \prod_{ij}\begin{cases} N(X_k[i, j]\mid F[i,j-d_k], s^2), & \text {if $[i, j]\in faceArea(d_k)$} \\\\ N(X_k[i, j]\mid B[i, j], s^2), & \text{otherwise} \end{cases} $$
```
def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k, F, B, s) for all images X_k in X and
    all possible face position d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(W-w+1, K)
        ll[dw, k] - log-likelihood of observing image X_k given
        that the prankster's face F is located at position dw
    """
    # your code here
    H, W, K = np.shape(X)
    _, w = np.shape(F)
    ll = np.zeros((W-w+1, K), dtype=float)
    for dw in range(W-w+1):
        for k in range(K):
            ll[dw, k] = H*w*np.log(1/(s*np.sqrt(2*np.pi))) - np.sum((X[:, dw:dw+w, k] - F)**2/(2 * s**2)) + \
            H*dw*np.log(1/(s*np.sqrt(2*np.pi))) - np.sum((X[:, 0:dw, k] - B[:, 0:dw])**2/(2 * s**2)) + \
            H*(W-w-dw)*np.log(1/(s*np.sqrt(2*np.pi))) - np.sum((X[:, dw+w:, k] - B[:, dw+w:])**2/(2 * s**2))
    return ll
```
* 实现Variational lower bound
 ```
 def calculate_lower_bound(X, F, B, s, a, q):
    """
    Calculates the lower bound L(q, F, B, s, a) for
    the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1)
        Estimate of prior on position of face in any image.
    q : array
        q[dw, k] - estimate of posterior
                   of position dw
                   of prankster's face given image Xk

    Returns
    -------
    L : float
        The lower bound L(q, F, B, s, a)
        for the marginal log likelihood.
    """
    # your code here
    ll = calculate_log_probability(X, F, B, s)
    L = np.sum(q*ll) + np.sum(q.T*np.log(a)) - np.sum(q*np.log(q))
    return L
    ```
* E步
```
def run_e_step(X, F, B, s, a):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k, F, B, s, a).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(H, w)
        Estimate of prankster's face.
    B : array shape(H, W)
        Estimate of background.
    s : float
        Eestimate of standard deviation of Gaussian noise.
    a : array, shape(W-w+1)
        Estimate of prior on face position in any image.

    Returns
    -------
    q : array
        shape (W-w+1, K)
        q[dw, k] - estimate of posterior of position dw
        of prankster's face given image Xk
    """
    # your code here
    # 使用logsumexp的方法
    # ad = log p(x_k| d_k, F, B, s) + log p(d_k,|a)
    # d* = argmax_d{ad}
    # log \sum exp(ad)= ad* + log\sum_d exp(ad-ad*)
    ll = calculate_log_probability(X, F, B, s)
    ad = ll + np.log(a).T.reshape(a.shape[0], 1)
    ad_max = np.max(ad, axis=0)
    q = ad - ad_max - np.log(np.sum(np.exp(ad - ad_max), axis=0))
    q = np.exp(q)
    return q
```
* M步
```
def run_m_step(X, q, w):
    """
    Estimates F, B, s, a given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    q  :
        q[dw, k] - estimate of posterior of position dw
                   of prankster's face given image Xk
    w : int
        Face mask width.

    Returns
    -------
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1)
        Estimate of prior on position of face in any image.
    """
    # your code here
    H, W, K = X.shape
    F = np.zeros((H, w))
    B = np.zeros((H, W))
    s = 0.0
    # a
    a = np.sum(q, axis=1)/np.sum(q)
    # F
    for m in range(w):
        for k in range(K):
            F[:, m] = (1/K*np.sum(q[:, k]*X[:, m:W-w+1+m, k], axis=1)) + F[:, m]

    # B
    B1 = np.zeros((H, W))
    B2 = np.zeros((H, W))
    for dw in range(W-w+1):
        for k in range(K):
            B1[:, :dw] = q[dw, k] * X[:, :dw, k] + B1[:, :dw]
            B1[:, dw+w:] = q[dw, k] * X[:, dw+w:, k] + B1[:, dw+w:]
            B2[:, :dw] = q[dw, k] + B2[:, :dw]
            B2[:, dw+w:] = q[dw, k] + B2[:, dw+w:]

    B = B1/B2

    # s
    s_square = 0
    for dw in range(W-w+1):
        for k in range(K):
            s_square = q[dw, k] * (np.sum((X[:, :dw, k] - B[:, :dw])**2) + np.sum((X[:, dw:dw+w, k] - F)**2)+ \
            np.sum((X[:, dw+w:, k] - B[:, dw+w:])**2)) + s_square
    s = np.sqrt(1/(K*W*H) *  s_square)

    return F, B, s, a
```
* 对&E&步与&M&步交替运行，当$L(q, \,F, \,B, \,s, \,a)$增加值小于提前设定好的阈值时，程序结束。
```
def run_m_step(X, q, w):
    """
    Estimates F, B, s, a given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    q  :
        q[dw, k] - estimate of posterior of position dw
                   of prankster's face given image Xk
    w : int
        Face mask width.

    Returns
    -------
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1)
        Estimate of prior on position of face in any image.
    """
    # your code here
    H, W, K = X.shape
    F = np.zeros((H, w))
    B = np.zeros((H, W))
    s = 0.0
    # a
    a = np.sum(q, axis=1)/np.sum(q)
    # F
    for m in range(w):
        for k in range(K):
            F[:, m] = (1/K*np.sum(q[:, k]*X[:, m:W-w+1+m, k], axis=1)) + F[:, m]

    # B
    B1 = np.zeros((H, W))
    B2 = np.zeros((H, W))
    for dw in range(W-w+1):
        for k in range(K):
            B1[:, :dw] = q[dw, k] * X[:, :dw, k] + B1[:, :dw]
            B1[:, dw+w:] = q[dw, k] * X[:, dw+w:, k] + B1[:, dw+w:]
            B2[:, :dw] = q[dw, k] + B2[:, :dw]
            B2[:, dw+w:] = q[dw, k] + B2[:, dw+w:]

    B = B1/B2

    # s
    s_square = 0
    for dw in range(W-w+1):
        for k in range(K):
            s_square = q[dw, k] * (np.sum((X[:, :dw, k] - B[:, :dw])**2) + np.sum((X[:, dw:dw+w, k] - F)**2)+ \
            np.sum((X[:, dw+w:, k] - B[:, dw+w:])**2)) + s_square
    s = np.sqrt(1/(K*W*H) *  s_square)

    return F, B, s, a
```
最终实验结果，人脸图片和背景图如下所示：
![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/EM%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0/%E4%BA%BA%E8%84%B81.png)  ![](https://raw.githubusercontent.com/hjyai94/Blog/master/source/uploads/EM%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0/%E8%83%8C%E6%99%AF1.png)

# 问题与总结
变成中遇到了一个问题，就是求解q时会出现0的情况，这样就会导致$log\ 0$的情况出现，出现错误，不知道怎么结果，但是结果却可以跑出了，不过图片并不是特别清晰，我想如果用deep learning的方法训练估计可以达到一个比较好的结果。EM算法可以看成特殊的坐标下降发，对lower bound按照隐变量和参数进行坐标寻优。

# 参考资料
[1] http://deepbayes.ru/ 主要来自其中的slide
[2] https://cw.fel.cvut.cz/old/courses/ae4b33rpz/labs/12_em/start
