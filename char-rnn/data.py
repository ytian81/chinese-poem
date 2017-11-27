#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import json
import re
import torch

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

if __name__ == '__main__':
  load_poem(1000, '李白', constraint=7)
