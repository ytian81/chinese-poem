#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import json
import re
import torch
import jieba
import jieba.posseg
import collections
import numpy as np
from sklearn.decomposition import PCA

from utils import Variable

BASE_DIR = '../dataset/chinese-poetry/json/poet.*.*.json'

def is_constrained(paragraphs, constraint):
    parts = re.split("[，！。？]", paragraphs)
    for part in parts:
        if part == "":
            continue
        if len(part) != constraint:
            return False
    return True

def load_poem(max_train=None, author=None, constraint=None, base_dir=BASE_DIR):
  poems = []
  for file in glob.glob(base_dir):
    try:
      with open(file, encoding='utf-8') as f:
        poems.extend(json.loads(f.read()))
    except:
      print("{} doesn't exist".format(file))
      raise

  for poem in poems:
    poem["paragraphs"] = clean_poem(poem["paragraphs"])

  if author is not None:
    poems = [poem for poem in poems if poem["author"]==author]

  if constraint is not None:
    poems = [poem for poem in poems if is_constrained(poem["paragraphs"], constraint)]

  if max_train is not None and len(poems) > max_train:
    poems = poems[:max_train]

  return poems

def clean_poem(poem):
  poem = ''.join(poem)
  poem = re.sub("（.*）", "", poem)
  poem = re.sub("{.*}", "", poem)
  poem = re.sub("《.*》", "", poem)
  poem = re.sub("[\]\[]", "", poem)
  res = ""
  for s in poem:
    if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
      res += s
  res = re.sub("。。", "。", res)
  return res

def make_paragraph(poems):
  poems = [poem["paragraphs"] for poem in poems if poem["paragraphs"] != '']
  return poems

def make_vocab(poems):
  if len(poems)>1:
    poems = "".join(poems)
  vocab = set([word for word in poems ])
  vocab = list(vocab)
  vocab.append('<EOP>')
  return vocab, len(vocab)

def poem_to_tensor(poem, vocab, is_target=False):
  word_indexes = [vocab.index(word) for word in poem]
  if is_target:
    word_indexes.append(vocab.index('<EOP>'))
  return Variable(torch.LongTensor(word_indexes))

def high_freq_word_cut(poems):
  poems = ''.join(poems)
  words = jieba.posseg.cut(poems)
  seg_list = (word for word, tag in words if tag.startswith('n'))
  count = collections.Counter(seg_list)
  return dict(count)

def make_sub_topics(count, poems):
  count_keys = list(count.keys())

  # build sub_topics
  sub_topics = np.zeros((len(poems), len(count)))
  for idx, poem in enumerate(poems):
      words = jieba.posseg.cut(poem)
      seg_list = (word for word, tag in words if tag.startswith('n'))
      seg_list_index = [count_keys.index(key) for key in seg_list]
      sub_topics[idx, seg_list_index] = 1
  return sub_topics

def make_PCA_reduction(sub_topics, embedding_size):
  pca = PCA(n_components=embedding_size)
  reduced_sub_topics = pca.fit_transform(sub_topics)
  return reduced_sub_topics, pca

if __name__ == '__main__':
  load_poem(1000, '李白', constraint=7)
