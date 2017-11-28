#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import torch
from hanziconv import HanziConv

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)
        if USE_CUDA:
            self.data = self.data.cuda()

def time_since(since):
  now = time.time()
  s = now - since
  m = math.floor(s / 60)
  s -= m * 60
  return "%dm %ds" % (m, s)

def simplify(func):
  def wrapper(*args, **kwargs):
    predicted = func(*args, **kwargs)
    return HanziConv.toSimplified(predicted)
  return wrapper