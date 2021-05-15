---
title: 降维-概率PCA
date: 2020-01-02 16:45:38
tags: 机器学习理论推导
categories: 学习
---
# 定义
$$x \in \mathbb{R}^p, \quad z \in \mathbb{R}^q, \quad q< p $$   $$x:\ \text{observed data}, z:\ \text{latent variable} $$

线性高斯模型：
\begin{cases}
z \sim N\left(0, I_q\right) \\\\
x=wz+\mu+\varepsilon \\\\
\varepsilon \sim N\left(0, \sigma^{2} I_{p}\right)
\end{cases}

求$z, x|z, x, z|x$

1. $x|z$
$$
\begin{aligned}
&E[x | z]=E[w z+\mu+\varepsilon]=w z+\mu\\\\
&\operatorname{Var}[x | z]=\operatorname{Var}[w z+\mu+\varepsilon]=\sigma^{2} I\\\\
&x | z \sim N\left(wz+\mu, \sigma^{2} I\right)
\end{aligned}
$$

2. $z|x$
\begin{aligned}
\operatorname{cov}(x, z) &=E\left[(x-\mu)(z-0)^{T}\right] \\\\
&=E\left[(x-\mu) z^T \right]\\\\
&=E\left[(w z+\varepsilon) z^{T}\right] \\\\
&=E\left[w z z^{\top}+\varepsilon z^T \right] \\\\
&= wE\left[zz^T\right] + E \left[ \varepsilon \right] E \left[ z^T \right] \\\\
&= w
\end{aligned}

\begin{aligned}
E[x]&=E\left[w z+\mu+\varepsilon\right]=E\left[w z+\mu\right]+E[\varepsilon]=\mu \\\\
\operatorname{Var}[x] &=\operatorname{Var}\left[w z+\mu+\varepsilon\right]=\operatorname{Var}[w z]+Var [\varepsilon]\\\\
&= w \cdot I \cdot w^{\top}+\sigma^{2} I=w w^{\top}+\sigma^{2} I
\end{aligned}

$$
\left(\begin{array}{l}
{x} \\\\
{z}
\end{array}\right) \sim N\left( \left[\begin{array}{l}
{\mu} \\\\
{0}
\end{array}\right],\left[\begin{array}{ll}
{ww^T + \sigma^2 I} & {w} \\\\
{w^T} & {I}
\end{array}\right] \right)
$$

直接套用补充的公式可以得到 $z|x$的分布。






# 补充
$$x=\left(\begin{array}{l}{x_{a}} \\\\ {x_{b}}\end{array}\right), x \sim N\left(\left[\begin{array}{l}{\mu_{a}} \\\\ {\mu_{b}}\end{array}\right],\left[\begin{array}{ll}{\Sigma_{aa}} & {\Sigma_{a b}} \\\\ {\Sigma_{ba}} & {\Sigma_{b b}}\end{array}\right]\right)$$

$$x_{b\cdot a}=x_{b}-\Sigma_{b a} \Sigma_{a a}^{-1} x_{a}$$ $$\mu_{b\cdot a}=\mu_{b}-\Sigma_{b a} \Sigma_{a a}^{-1} \mu_{a}$$ $$\Sigma_{b b\cdot a}=\Sigma_{b b}-\Sigma_{b a} \Sigma_{a a}^{-1} \Sigma_{a b} \quad$$  $$x_{b\cdot a} \sim N\left(\mu_{b a}, \Sigma_{bb\cdot a} \right)$$

又$x_{b}=x_{b\cdot a}+\Sigma_{b a} \Sigma_{a a}^{-1} x_{a}$

$$
\begin{aligned}
&E\left[x_{b} | x_{a}\right]=\mu_{b \cdot a} + \Sigma_{b a} \Sigma_{a a}^{-1} x_{a}\\\\
&\operatorname{Var}\left[x_{b} | x_{a}\right]=\operatorname{Var}\left[x_{b\cdot a} \right]=\Sigma_{b b \cdot a}
\end{aligned}
$$
$$ x_b | x_a \sim N \left(\mu_{b \cdot a} + \Sigma_{b a} \Sigma_{a a}^{-1} x_{a}, \Sigma_{b b \cdot a} \right) $$
