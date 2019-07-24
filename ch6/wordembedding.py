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
from QACNN.utils import trasformText
from gensim.models import word2vec

# 引入日志配置
import logging
import cPickle as pkl
import os
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

processed_data_path = './data/WikiQA/processed'
with open(os.path.join(processed_data_path, 'vocab.pkl'), 'r') as fr:
        wordDic,id2word = pkl.load(fr)
inList=[20935, 15080, 11093, 22327, 25169, 3165, 11104]
print(trasformText(inList,id2word))