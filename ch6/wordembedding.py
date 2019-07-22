#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# Author: thunder
# Date: 2019-07-22:18:56:20
#
#===============================================================================

from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 切分词汇
sentences= [s.encode('utf-8').split() for s in sentences]

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1,workers=4)
print(model.most_similar(['quick']))
# 进行相关性比较
print(model.similarity('dogs','fox'))