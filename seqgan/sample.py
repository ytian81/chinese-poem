import torch
import pickle as p
from utils import *

model = torch.load('generator_pretrain.pt')
max_length = 200
rFile = open('wordDic', 'rb')
word_to_ix = p.load(rFile)

print(list(model.parameters())[0])


ix_to_word = invert_dict(word_to_ix)


# Sample from a category and starting letter
def sample(startWords=['<START>']):
    input = make_one_hot_vec_target(startWords[0], word_to_ix)
    hidden = model.initHidden()
    output_name = ""
    if (startWords[0] != "<START>"):
        output_name = startWords[0]
    for i in range(max_length):
        output, hidden = model(input, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        w = ix_to_word[topi]
        if i < len(startWords)-1:
            w = startWords[i+1]
        if w == "<END>" or w == "<PAD>":
            break
        else:
            output_name += w
        input = make_one_hot_vec_target(w, word_to_ix)
    return output_name



print(sample(['<START>',"春"]))
#print(sample(['<START>',"花"]))
#print(sample(['<START>',"秋"]))
#print(sample(['<START>',"明","日"]))
#print(sample(['<START>',"君"]))
print(sample())