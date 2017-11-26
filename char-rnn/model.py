#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from utils import Variable

class RNN(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, lstm_size=2):
    super(RNN, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.lstm_size = lstm_size

    self.encoder = nn.Embedding(vocab_size, embedding_size)
    self.lstm = nn.LSTM(embedding_size, hidden_size, lstm_size)
    self.decoder = nn.Linear(hidden_size, vocab_size)
    # TODO:
    # add dropout

  def forward(self, word, hidden):
    embedding = self.encoder(word.view(1, -1))
    output, hidden = self.lstm(embedding.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.lstm_size, 1, self.hidden_size).zero_()),
            Variable(weight.new(self.lstm_size, 1, self.hidden_size).zero_()))