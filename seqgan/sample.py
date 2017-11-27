import torch
import pickle as p
from utils import *

model = torch.load('generator_pretrain.pt')
max_length = 200
rFile = open('wordDic', 'rb')
word_to_ix = p.load(rFile)

def invert_dict(d):
    return dict((v, k) for k, v in d.items())


ix_to_word = invert_dict(word_to_ix)


# Sample from a category and starting letter
def sample(startWord='<START>'):
    input = make_one_hot_vec_target(startWord, word_to_ix)
    hidden = model.initHidden()
    output_name = ""
    if (startWord != "<START>"):
        output_name = startWord
    for i in range(max_length):
        output, hidden = model(input, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        w = ix_to_word[topi]
        if w == "<END>":
            break
        else:
            output_name += w
        input = make_one_hot_vec_target(w, word_to_ix)
    return output_name



print(sample("春"))
print(sample("花"))
print(sample("秋"))
print(sample("月"))