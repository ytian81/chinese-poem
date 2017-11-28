import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class GeneratorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(GeneratorModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        batch_size = input.size()[0]

        embeds = self.embeddings(input).view((1, batch_size, -1))
        output, hidden = self.lstm(embeds, hidden)
        output = F.relu(self.linear1(output.view(batch_size, -1)))

        output = self.softmax(output)
        return output, hidden

    def initHidden(self, length=1, batch_size=1):
        return (Variable(torch.zeros(length, batch_size, self.hidden_dim)),
                Variable(torch.zeros(length, batch_size, self.hidden_dim)))

    def NLLLoss(self, input, target):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = input.size()

        input = input.permute(1, 0)  # seq_len x batch_size
        target = target.permute(1, 0)  # seq_len x batch_size
        h = self.initHidden(batch_size=batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(input[i], h)
            loss += loss_fn(out, target[i])

        return loss  # per batch

    def PGLoss(self, input, target, reward):
        batch_size, seq_len = input.size()
        input = input.permute(1, 0)  # seq_len x batch_size
        target = target.permute(1, 0)  # seq_len x batch_size
        h = self.initHidden(batch_size=batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(input[i], h)
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]] * reward[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss / batch_size


class DiscriminatorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len, filter_sizes, linear_hid_sizes):
        super(DiscriminatorModel, self).__init__()
        self.filter_sizes = filter_sizes
        self.seq_len = seq_len
        self.linear_hid_sizes = linear_hid_sizes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        current_size = [embedding_dim, seq_len]
        current_nfilter = 1
        self.filters = []
        for i in range(len(filter_sizes)):
            self.filters.append(nn.Sequential(
                nn.Conv2d(current_nfilter, int(filter_sizes[i][0]), kernel_size=int(filter_sizes[i][1]), padding=int(np.ceil(filter_sizes[i][1]/2)), stride=int(filter_sizes[i][2])),
                nn.BatchNorm2d(filter_sizes[i][0]),
                nn.ReLU(),
                nn.MaxPool2d(2)))
            current_size[0] = np.ceil((current_size[0])/filter_sizes[i][2]/2)
            current_size[1] = np.ceil((current_size[1])/filter_sizes[i][2]/2)
            current_nfilter = filter_sizes[i][0]
        self.linears = []
        last_dim = 88#current_size[0]*current_size[1]*current_nfilter
        for i in range(len(linear_hid_sizes)):
            self.linears.append(nn.Linear(int(last_dim), int(self.linear_hid_sizes[i])))
            last_dim = self.linear_hid_sizes[i]

        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, seq_len = input.size()

        embeds = self.embeddings(input).view((seq_len, batch_size, -1))
        lval = embeds.view(batch_size, 1, seq_len, -1)
        for i in range(len(self.filters)):
            lval = self.filters[i](lval)
        lval = lval.view(batch_size, -1)

        for i in range(len(self.linear_hid_sizes)):
            lval = F.tanh(self.linears[i](lval))
            lval = self.dropout(lval)
        output = self.sigmoid(lval)

        return output


    def batchClassify(self, input):
        out = self.forward(input)
        return out.view(-1)

    '''def __init__(self, vocab_size, embedding_dim, lstm_hid_sizes, linear_hid_sizes):
        super(DiscriminatorModel, self).__init__()
        self.lstm_hid_sizes = lstm_hid_sizes
        self.linear_hid_sizes = linear_hid_sizes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        last_dim = embedding_dim
        self.lstms = []
        for i in range(len(lstm_hid_sizes)):
            self.lstms.append(nn.LSTM(last_dim, self.lstm_hid_sizes[i]))
            last_dim = self.lstm_hid_sizes[i]
        self.linears = []
        for i in range(len(linear_hid_sizes)):
            self.linears.append(nn.Linear(last_dim, self.linear_hid_sizes[i]))
            last_dim = self.linear_hid_sizes[i]

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        batch_size, seq_len = input.size()

        hiddens = self.initHidden(batch_size=batch_size)
        embeds = self.embeddings(input).view((seq_len, batch_size, -1))

        lval = embeds
        new_hidden = []
        for i in range(len(self.lstm_hid_sizes)):
            lval, hidden = self.lstms[i](lval, hiddens[i])
            new_hidden.append(hidden)
        lval = lval[-1].view(batch_size, -1)
        for i in range(len(self.linear_hid_sizes)):
            lval = F.relu(self.linears[i](lval))
            lval = self.dropout(lval)
        output = self.softmax(lval)

        return output, hiddens

    def initHidden(self, length=1, batch_size=1):
        hiddens = []
        for dim in self.lstm_hid_sizes:
            hiddens.append((Variable(torch.zeros(length, batch_size, dim)),
                Variable(torch.zeros(length, batch_size, dim))))
        return hiddens

    def batchClassify(self, input):
        out, hiddens = self.forward(input)
        return out'''

