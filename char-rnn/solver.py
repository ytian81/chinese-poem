#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
import os

from utils import Variable, time_since, USE_CUDA, simplify
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

    self.save_every = kwargs.pop('save_every', 1000)
    self.print_every = kwargs.pop('print_every', 100)
    self.plot_every = kwargs.pop('plot_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError('Unrecognized arguments %s' % extra)

    self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.criterion = nn.CrossEntropyLoss()
    self.all_losses = []

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
    prefix = "model/"
    prefix += time.strftime('%m-%d_%H-%M/', time.localtime(start))
    if not os.path.exists(prefix):
      os.makedirs(prefix)

    best_loss = float('inf')
    best_model = self.model

    loss_avg = 0.

    for epoch in range(1, self.num_epochs+1):
      loss = self.train_one_step(*self.sample())
      loss_avg += loss

      if loss < best_loss:
        best_loss = loss
        best_model = self.model

      if epoch % self.print_every == 0 and self.verbose:
        print('[%s, %d%%, %.4f]' % (time_since(start), epoch / self.num_epochs * 100, loss))
        print(self.evaluate())

      if epoch % self.plot_every == 0:
        self.all_losses.append(loss_avg / self.plot_every)
        loss_avg = 0

      if epoch % self.save_every == 0:
        model_name = prefix
        model_name += "char_rnn_%d.model" % epoch
        torch.save(self.model, model_name)

    model_name = prefix
    model_name += "best_char_rnn.model"
    torch.save(best_model, model_name)
    print("save best model with loss %.3f" % best_loss)

  @simplify
  def evaluate(self, start_with="牀前看月光", temperature=0.8, max_length=1000):
    hidden = self.model.init_hidden()
    self.model.zero_grad()
    start_input = poem_to_tensor(start_with, self.vocab)
    predicted = start_with

    # prepare hidden state
    for input_word in start_input[:-1]:
      _, hidden = self.model(input_word, hidden)

    word = input_word[-1]
    for _ in range(max_length):
      output, hidden = self.model(word, hidden)

      # sample from network 
      output_dist = output.data.view(-1).div(temperature).exp()
      top_index = torch.multinomial(output_dist, 1)[0]

      predicted_word = self.vocab[top_index]
      if predicted_word == '<EOP>':
        break
      predicted += predicted_word
      word = poem_to_tensor(predicted_word, self.vocab)

    return predicted