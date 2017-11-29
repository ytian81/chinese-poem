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
    args['gen_pretrain_epochs'] = 20

    args['dis_retrain'] = True
    args['dis_step_num'] = 5
    args['dis_epoch_num'] = 5
    args['dis_neg_sample_num'] = 1000
    args['dis_batch_size'] = 32

    args['adv_epochs'] = 5
    args['adv_pg_iters'] = 15
    args['adv_pg_samples'] = 200
    args['use_cem'] = False
    args['cem_std'] = None

    args['use_gpu'] = False

    return args

def pre_process_data():
    data = dataParser.parseRawData(constrain = 5, author="李白", max_len=60)

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
        if len(sent) > 500:
            print(sent)

    VOCAB_SIZE = len(word_to_ix)

    print("VOCAB_SIZE:", VOCAB_SIZE)
    print("data_size", len(data))

    for i in range(len(data)):
        data[i] = toList(data[i])
        data[i].append("<END>")

    p.dump(word_to_ix, open('wordDic', 'wb'))

    vec_data = prepare_batch_sequence(np.array(data), word_to_ix)

    return VOCAB_SIZE, word_to_ix, vec_data

def train_generator(generator, true_data, args, use_gpu = False):
    optimizer=optim.Adam(generator.parameters(), lr=1e-2)
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
    return generator

def train_discriminator(discriminator, generator, true_data, word_to_ix, args, use_gpu=False, step_num=10, epoch_num=3):
    optimizer = optim.Adagrad(discriminator.parameters())
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

            ss = batchwise_sample(generator, 4, 2, len(true_data[0]), word_to_ix)
            dis_inps, dis_targets = prepare_discriminator_training_data(true_data[0:4], ss, gpu=use_gpu)
            print(discriminator.batchClassify(dis_inps))
    return discriminator


def train_generator_PG(discriminator, generator, word_to_ix, true_data, args, use_gpu=False):
    optimizer = optim.Adam(generator.parameters(), lr=1e-2)
    for i in range(args['adv_pg_iters']):
        print("-----Adversarial PG Iter ", i, " -----------")
        s = batchwise_sample(generator, args['adv_pg_samples'], 1, len(true_data[0]), word_to_ix)
        input, target = prepare_generator_training_data(s, gpu=use_gpu)
        rewards = discriminator.batchClassify(input)

        optimizer.zero_grad()
        pg_loss = generator.PGLoss(input, target, rewards)
        pg_loss.backward()
        optimizer.step()
        print('PG_loss\t%f\tAvgReward\t%f'%(pg_loss.data[0],float(np.mean(rewards.data.numpy()))))
    torch.save(generator, 'generator_adv.pt')
    return generator



def train_generator_CEM(discriminator, generator, word_to_ix, true_data, args, use_gpu=False):
    eval_samp_num = 4
    param_samp_num = int(args['adv_pg_samples'] / eval_samp_num)

    generator_parameters = list(generator.parameters())
    flat_pm = param2flat(generator_parameters)

    if args['cem_std'] is None:
        args['cem_std'] = np.ones(len(flat_pm)) * np.std(flat_pm)

    for i in range(args['adv_pg_iters']):
        flat_pm = param2flat(generator_parameters)

        samp_params = []
        for j in range(param_samp_num):
            samp_params.append(np.array(np.random.normal(0, 1, len(flat_pm))*args['cem_std'] + flat_pm, dtype=np.float32))

        avg_rewards = []
        for j in range(param_samp_num):
            flat2param(samp_params[j], generator_parameters)

            s = batchwise_sample(generator, eval_samp_num, 1, len(true_data[0]), word_to_ix)
            input, target = prepare_generator_training_data(s, gpu=use_gpu)
            rewards = discriminator.batchClassify(input)
            avg_rewards.append(np.mean(rewards.data.numpy()))
        sorted_rwds = np.copy(avg_rewards)
        sorted_rwds.sort()
        max_k_val = sorted_rwds[int(-param_samp_num/2.0)]

        selected_params = []
        for j in range(param_samp_num):
            if avg_rewards[j] > max_k_val:
                selected_params.append(samp_params[j])
        mean_param = np.array(np.mean(selected_params, axis=0), dtype=np.float32)
        std_param = np.std(selected_params, axis=0)
        flat2param(mean_param, generator_parameters)
        args['cem_std'] = std_param

        print('AvgReward\t%f' % (float(np.mean(avg_rewards))))
    torch.save(generator, 'generator_adv_cem.pt')
    return generator




def train_adversarial(discriminator, generator, true_data, word_to_ix, args, use_gpu=False):
    for epoch in range(args['adv_epochs']):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator')
        if args['use_cem']:
            generator = train_generator_CEM(discriminator, generator, word_to_ix, true_data, args)
        else:
            generator = train_generator_PG(discriminator, generator, word_to_ix, true_data, args)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        discriminator = train_discriminator(discriminator, generator, true_data, word_to_ix, args, args['use_gpu'], step_num=5, epoch_num=3)
    return generator

def main():
    args = prepare_args()
    # pre-process the poem data
    vocab_size, word_to_ix, vec_data = pre_process_data()

    # Initialize variables
    generator = GeneratorModel(vocab_size, 256, 256)
    discriminator = DiscriminatorModel(vocab_size, 512, len(vec_data[0]), [[16, 3, 1], [8, 5, 2], [4, 7, 3]], [128, 64, 1])

    # pre-training using maximum-likelihood model or load it directly
    print('\n=========\nPre-train Generator \n=========')
    if args['gen_retrain']:
        generator = train_generator(generator, vec_data, args, args['use_gpu'])
        torch.save(generator, 'generator_pretrain.pt')
    else:
        generator = torch.load('generator_pretrain.pt')

    # pre-train discriminator model
    print('\n=========\nPre-train Discriminator \n=========')
    if args['dis_retrain']:
        discriminator = train_discriminator(discriminator, generator, vec_data, word_to_ix, args, args['use_gpu'], step_num=args['dis_step_num'], epoch_num=args['dis_epoch_num'])
        torch.save(discriminator, 'discriminator_pretrain.pt')
    else:
        discriminator = torch.load('discriminator_pretrain.pt')

    # start adversarial training
    print('\n=========\nAdversarial Training \n=========')
    generator = train_adversarial(discriminator, generator, vec_data, word_to_ix, args, args['use_gpu'])
    if args['use_cem']:
        torch.save(generator, 'generator_adv_cem.pt')
    else:
        torch.save(generator, 'generator_adv.pt')



if __name__ == '__main__':
    main()