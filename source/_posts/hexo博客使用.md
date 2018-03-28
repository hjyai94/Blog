---
title: hexo博客使用
date: 2017-11-22 10:51:49
tags:
- hexo
categories: 技术
---
# 简单的语句
hexo clean 清理缓存
hexo g 或者 hexo generate 产生文件
hexo d 或者 hexo deploy 发布博客

# 重复输入密码问题
启动ssh-agent后台 `eval $(ssh-agent)`
添加密码 `ssh-add`
这样就可以不必在当前的terminal重复输入密码了。
