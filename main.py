#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io

import data 

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='un', help='name of corpus')
parser.add_argument('--data_path', type=str, default='un/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='skipgram/embeddings.txt', help='directory containing embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')
parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

args = parser.parse_args()

pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

## get data
# 1. vocabulary
print('Getting vocabulary ...')
data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))
vocab, train_data_with_time, validation_data, test_1_data, test_2_data, test_data = data.get_data()
vocab_size = len(vocab)
args.vocab_size = vocab_size

# 1. training data
print('Getting training data ...')

train_data, train_times = data.get_time_columns(train_data_with_time)
args.num_times = len(np.unique(train_times))
train_rnn_inp = data.get_rnn_input(train_data_with_time, args.num_times, args.vocab_size)

# 2. dev set
print('Getting validation data ...')
valid_rnn_inp = data.get_rnn_input(validation_data, args.num_times, args.vocab_size)

# 3. test data
print('Getting testing data ...')

test_rnn_inp = data.get_rnn_input(test_data, args.num_times, args.vocab_size)

test_1_rnn_inp = data.get_rnn_input(test_1_data, args.num_times, args.vocab_size)
test_2_rnn_inp = data.get_rnn_input(test_2_data, args.num_times, args.vocab_size)

embeddings = None
if not args.train_embeddings:
    embeddings = data.read_embedding_matrix(vocab, device, load_trainned=False)
    args.embeddings_dim = embeddings.size()

print('\n')
print('=*'*100)
print('Training a Dynamic Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'detm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_L_{}_minDF_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.eta_nlayers, args.min_df, args.train_embeddings))

## define model and optimizer
if args.load_from != '':
    print('Loading checkpoint from {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
else:
    model = DETM(args, embeddings)
print('\nDETM architecture: {}'.format(model))
model.to(device)
if args.mode == 'train':
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    for epoch in range(1, args.epochs):
        model.train_for_epoch(epoch, args, )
        if epoch % args.visualize_every == 0:
            model.visualize(args, vocab)
        val_ppl = model.get_completion_ppl('val', args, validation_data, test_1_data, test_2_data)
        print('val_ppl: ', val_ppl)
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = model.optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                model.optimizer.param_groups[0]['lr'] /= args.lr_factor
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        print('saving topic matrix beta...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().numpy()
        scipy.io.savemat(ckpt+'_beta.mat', {'values': beta}, do_compression=True)
        if args.train_embeddings:
            print('saving word embedding matrix rho...')
            rho = model.rho.weight.cpu().numpy()
            scipy.io.savemat(ckpt+'_rho.mat', {'values': rho}, do_compression=True)
        print('computing validation perplexity...')
        val_ppl = model.get_completion_ppl('val', args, validation_data, test_1_data, test_2_data)
        print('computing test perplexity...')
        test_ppl = model.get_completion_ppl('test', args, validation_data, test_1_data, test_2_data)
else:
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    print('saving alpha...')
    with torch.no_grad():
        alpha = model.mu_q_alpha.cpu().numpy()
        scipy.io.savemat(ckpt+'_alpha.mat', {'values': alpha}, do_compression=True)

    print('computing validation perplexity...')
    val_ppl = model.get_completion_ppl('val', args, validation_data, test_1_data, test_2_data)
    print('computing test perplexity...')
    test_ppl = model.get_completion_ppl('test', args, validation_data, test_1_data, test_2_data)
    print('computing topic coherence and topic diversity...')
    model.get_topic_quality(args, train_data)
    print('visualizing topics and embeddings...')
    model.visualize(args, vocab)
