import torch
import torch.optim as optim
import pickle as p
from model import *
import dataParser
from utils import *

def prepare_args():
    args = {}
    args['gen_retrain'] = True
    args['gen_batch_size'] = 32
    args['gen_pretrain_epochs'] = 15

    args['dis_retrain'] = True
    args['dis_step_num'] = 5
    args['dis_epoch_num'] = 5
    args['dis_neg_sample_num'] = 1000
    args['dis_batch_size'] = 32

    args['adv_epochs'] = 5
    args['adv_pg_iters'] = 15
    args['adv_pg_samples'] = 2000

    args['use_gpu'] = False

    return args

def pre_process_data():
    data = dataParser.parseRawData(max_len=33)

    datalen = [len(x) for x in data]
    print (np.unique(datalen))

    word_to_ix = {}

    word_to_ix['<START>'] = len(word_to_ix)
    word_to_ix['<END>'] = len(word_to_ix)
    word_to_ix['<PAD>'] = len(word_to_ix)

    for sent in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    VOCAB_SIZE = len(word_to_ix)

    print("VOCAB_SIZE:", VOCAB_SIZE)
    print("data_size", len(data))

    for i in range(len(data)):
        data[i] = toList(data[i])
        data[i].append("<END>")

    p.dump(word_to_ix, open('wordDic', 'wb'))

    vec_data = prepare_batch_sequence(np.array(data), word_to_ix)

    return VOCAB_SIZE, word_to_ix, vec_data

def train_generator(generator, optimizer, true_data, args, use_gpu = False):
    for epoch in range(args['gen_pretrain_epochs']):
        print('epoch %d : ' % (epoch + 1))
        total_loss = 0

        for i in range(0, len(true_data), args['gen_batch_size']):
            inp, target = prepare_generator_training_data(true_data[i:i + args['gen_batch_size']], gpu=args['use_gpu'])
            optimizer.zero_grad()
            loss = generator.NLLLoss(inp, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

        # each loss in a batch is loss per sample
        total_loss = total_loss / np.ceil(len(true_data) / float(args['gen_batch_size'])) / true_data.size()[1]
        print('Loss: ', total_loss)

def train_discriminator(discriminator, generator, optimizer, true_data, word_to_ix, args, use_gpu=False, step_num=10, epoch_num=3):
    for d_step in range(step_num):
        s = batchwise_sample(generator, args['dis_neg_sample_num'], args['dis_batch_size'], len(true_data[0]), word_to_ix)
        dis_inp, dis_target = prepare_discriminator_training_data(true_data, s, gpu=use_gpu)
        for epoch in range(epoch_num):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1))
            total_loss = 0
            total_acc = 0
            total_posneg = 0

            for i in range(0, len(true_data) + args['dis_neg_sample_num'], args['dis_batch_size']):
                inp, target = dis_inp[i:i + args['dis_batch_size']], dis_target[i:i + args['dis_batch_size']]
                optimizer.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.data[0]
                total_acc += torch.sum((out>0.5)==(target>0.5)).data[0]
                total_posneg += torch.sum((out<0.5)==(target<0.5)).data[0]

            total_loss /= np.ceil((len(true_data) + args['dis_neg_sample_num']) / float(args['dis_batch_size']))
            total_acc /= float(len(true_data) + args['dis_neg_sample_num'])
            print('Total loss and acc: ', total_loss, total_acc, total_posneg/float(len(true_data) + args['dis_neg_sample_num']))


def train_generator_PG(discriminator, generator, gen_optimizer, word_to_ix, true_data, args, use_gpu=False):
    for i in range(args['adv_pg_iters']):
        s = batchwise_sample(generator, args['adv_pg_samples'], 1, len(true_data[0]), word_to_ix)
        input, target = prepare_generator_training_data(s, gpu=use_gpu)
        rewards = discriminator.batchClassify(input)

        gen_optimizer.zero_grad()

        pg_loss = generator.PGLoss(input, target, rewards)
        pg_loss.backward()
        gen_optimizer.step()

        print('PG loss: ', pg_loss.data[0])


def train_adversarial(discriminator, generator, gen_optimizer, dis_optimizer, true_data, word_to_ix, args, use_gpu=False):
    for epoch in range(args['adv_epochs']):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator')
        train_generator_PG(discriminator, generator, gen_optimizer, word_to_ix, true_data, args)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(discriminator, generator, dis_optimizer, true_data, word_to_ix, args, args['use_gpu'], step_num=5, epoch_num=3)

def main():
    args = prepare_args()
    # pre-process the poem data
    vocab_size, word_to_ix, vec_data = pre_process_data()

    # Initialize variables
    generator = GeneratorModel(vocab_size, 256, 256)
    discriminator = DiscriminatorModel(vocab_size, 512, len(vec_data[0]), [[16, 3, 1], [8, 5, 2], [4, 7, 3]], [128, 64, 1])
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-2)
    dis_optimizer = optim.Adagrad(discriminator.parameters())

    # pre-training using maximum-likelihood model or load it directly
    print('\n=========\nPre-train Generator \n=========')
    if args['gen_retrain']:
        train_generator(generator, gen_optimizer, vec_data, args, args['use_gpu'])
        torch.save(generator, 'generator_pretrain.pt')
    else:
        generator = torch.load('generator_pretrain.pt')

    # pre-train discriminator model
    print('\n=========\nPre-train Discriminator \n=========')
    if args['dis_retrain']:
        train_discriminator(discriminator, generator, dis_optimizer, vec_data, word_to_ix, args, args['use_gpu'], step_num=args['dis_step_num'], epoch_num=args['dis_epoch_num'])
        torch.save(discriminator, 'discriminator_pretrain.pt')
    else:
        discriminator = torch.load('discriminator_pretrain.pt')

    # start adversarial training
    print('\n=========\nAdversarial Training \n=========')
    train_adversarial(discriminator, generator, gen_optimizer, dis_optimizer, vec_data, word_to_ix, args, args['use_gpu'])
    torch.save(generator, 'generator_adv.pt')



if __name__ == '__main__':
    main()