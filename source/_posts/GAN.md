---
title: GAN（生成对抗网络）
tags: GAN
categories: 学习
abbrlink: 26967
date: 2018-01-30 15:01:17
---
生成对抗网络（Generative Adersarial nets），基本思想就是同步训练生成模型G和判别模型D，
生成模型G服从数据分布，判别模型D用来判断样本是来自于生成模型G还是训练数据。


完全可以编成一段相声，从南边来了一个生成模型生成数据，从北边来了一个鉴别模型鉴别数据，
生成模型拼命生成数据不让鉴别模型鉴别数据，鉴别模型偏要鉴别生成模型生成的数据。

# Refercence
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
