#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import random
import time

from utils import Variable, time_since, USE_CUDA
from data import poem_to_tensor

class Solver(object):
  def __init__(self, model, data, vocab, **kwargs):
    super(Solver, self).__init__()
    self.model = model
    if USE_CUDA:
      self.model = self.model.cuda()
    self.data = data
    self.vocab = vocab

    # Unpack keyword arguments
    self.num_epochs = kwargs.pop('num_epochs', 2000)
    self.learning_rate = kwargs.pop('learning_rate', 1e-3)

    self.print_every = kwargs.pop('print_every', 100)
    self.plot_every = kwargs.pop('plot_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError('Unrecognized arguments %s' % extra)

    self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.criterion = nn.CrossEntropyLoss()

  def sample(self):
    datum = random.choice(self.data)
    input = poem_to_tensor(datum, self.vocab)
    target = poem_to_tensor(datum[1:], self.vocab, True)
    return input, target

  def train_one_step(self, input, target):
    hidden = self.model.init_hidden()
    self.model.zero_grad()
    loss = 0.

    for input_word, target_word in zip(input, target):
      output, hidden = self.model(input_word, hidden)
      loss += self.criterion(output, target_word)

    loss.backward()
    self.optim.step()

    return loss.data[0]/input.size()[0]

  def train(self):
    start = time.time()
    self.all_losses = []
    loss_avg = 0.

    for epoch in range(1, self.num_epochs+1):
      loss = self.train_one_step(*self.sample())
      loss_avg += loss

      if epoch % self.print_every == 0 and self.verbose:
        print('[%s, %d%%, %.4f]' % (time_since(start), epoch / self.num_epochs * 100, loss))

      if epoch % self.plot_every == 0:
        self.all_losses.append(loss_avg / self.plot_every)
        loss_avg = 0

  def evaluate(self, start_with, temperature=0.8, max_length=100):
    hidden = self.model.init_hidden()