import torch
import torch.autograd as autograd
import numpy as np

def make_one_hot_vec_target(word, word_to_ix):
    rst = autograd.Variable(torch.LongTensor([word_to_ix[word]]))
    return rst

def invert_dict(d):
    return dict((v, k) for k, v in d.items())

def prepare_batch_sequence(data, word_to_ix):
    max_len = np.max([len(s) for s in data])
    tensor_data = torch.zeros(len(data), int(max_len+2))
    for i in range(len(data)):
        idxs = [word_to_ix[w] for w in data[i]]
        idxs.insert(0, word_to_ix['<START>'])
        idxs.append(word_to_ix['<END>'])
        for j in range(len(idxs), max_len+2):
            idxs.append(word_to_ix['<END>'])

        tensor = torch.LongTensor(idxs)
        tensor_data[i] = tensor

    return tensor_data

def toList(sen):
    rst = []
    for s in sen:
        rst.append(s)
    return rst

def generate_samples(generator, max_length, sample_size, word_to_ix):
    samples = torch.zeros(sample_size, max_length)
    for i in range(sample_size):
        onesamp = []
        startWord = "<START>"
        input = make_one_hot_vec_target(startWord, word_to_ix)
        onesamp.append(word_to_ix[startWord])
        hidden = generator.initHidden()

        for j in range(max_length-1):
            output, hidden = generator(input, hidden)

            out = torch.multinomial(torch.exp(output), 1)
            #values, out = torch.max(output, 1)
            if out.data.numpy() == word_to_ix['<END>']:
                break
            onesamp.append(int(out.data.numpy()))
            input = out
        for j in range(len(onesamp), max_length):
            onesamp.append(word_to_ix['<PAD>'])

        tensor = torch.LongTensor(onesamp)
        samples[i] = tensor
    return samples

def prepare_generator_training_data(true_data, gpu=False):
    batch_size, seq_len = true_data.size()

    input = true_data[:, :seq_len-1]
    target = true_data[:, 1:]

    input = autograd.Variable(input).type(torch.LongTensor)
    target = autograd.Variable(target).type(torch.LongTensor)

    if gpu:
        input = input.cuda()
        target = target.cuda()

    return input, target

def prepare_discriminator_training_data(true_data, generate_data, gpu=False):
    input = torch.cat((true_data, generate_data), 0).type(torch.LongTensor)
    labels = torch.ones(true_data.size()[0] + generate_data.size()[0])
    labels[generate_data.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(labels.size()[0])
    input = input[perm]
    labels = labels[perm]

    input = autograd.Variable(input)
    labels = autograd.Variable(labels)

    if gpu:
        input = input.cuda()
        labels = labels.cuda()

    return input, labels

def idx2words(ids, word_to_ix):
    ix_to_word = invert_dict(word_to_ix)
    sent = ""
    idnum = ids.numpy()[0]
    print(idnum)
    for id in idnum:
        sent += ix_to_word[id]
    return sent

def batchwise_sample(generator, num_samples, batch_size, max_len, word_to_ix):
    samples = []
    for i in range(int(np.ceil(num_samples/float(batch_size)))):
        samples.append(generate_samples(generator, max_len, batch_size, word_to_ix))

    return torch.cat(samples, 0)[:num_samples]

# convert from list of parameter to 1d numpy array
def param2flat(params):
    pm = np.array([], dtype=np.float32)
    for p in params:
        pm = np.append(pm, p.data.numpy())
    return pm

def flat2param(flatpm, params):
    cur_pos = 0
    for pm in params:
        shape = pm.data.numpy().shape
        pm.data = torch.from_numpy(np.reshape(flatpm[cur_pos:cur_pos + np.prod(shape)], shape))







