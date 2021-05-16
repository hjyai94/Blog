---
title: 记录Ubuntu中出现的问题及解决方案
tags: Ubuntu
categories: 技术
abbrlink: 26865
date: 2018-06-14 00:00:00
---
# 执行sudo apt-get update 出错
出现错误如下：
`E: Some index files failed to download, they have been ignored, or old ones used instead'
出现无法更新的情况，可以在`/etc/apt/sources.list.d`中删除对应错误名字的文件。
