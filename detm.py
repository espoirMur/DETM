"""This file defines a dynamic etm object.
"""

import torch
import torch.nn.functional as F 
import numpy as np 
import math 
import data
from utils import nearest_neighbors

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DETM(nn.Module):
    def __init__(self, args, embeddings):
        super(DETM, self).__init__()

        ## define hyperparameters
        self.num_topics = args.num_topics
        self.num_times = args.num_times
        self.vocab_size = args.vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.eta_hidden_size = args.eta_hidden_size
        self.rho_size = args.rho_size
        self.emsize = args.emb_size
        self.enc_drop = args.enc_drop
        self.eta_nlayers = args.eta_nlayers
        self.t_drop = nn.Dropout(args.enc_drop)
        self.delta = args.delta
        self.train_embeddings = args.train_embeddings

        self.theta_act = self.get_activation(args.theta_act)

        ## define the word embedding matrix \rho
        if args.train_embeddings:
            self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(device)

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    self.theta_act,
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size)
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout)
        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def get_optimizer(self, args):
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'asgd':
            optimizer = torch.optim.ASGD(self.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)
        return optimizer

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2 ) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_eta(self, rnn_inp):
        ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times):
        """
        ## amortized inference
        Returns the topic proportions.
        """
        eta_td = eta[times.type('torch.LongTensor')]
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta

    def get_beta(self, alpha):
        """Returns the topic matrix \beta of shape K x V
        """
        if self.train_embeddings:
            logit = self.rho(alpha.view(alpha.size(0)*alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0)) 
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = loglik
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz
        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff

        beta = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff
        nelbo = nll + kl_alpha + kl_eta + kl_theta
        return nelbo, nll, kl_alpha, kl_eta, kl_theta

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))

    def train_for_epoch(self, epoch, args, train_data):
        """Train DETM on data for one epoch.
        """
        self.train()
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0
        indices = torch.randperm(args.num_docs_train)
        indices = torch.split(indices, args.batch_size)
        optimizer = self.get_activation(args.optimizer)
        for idx, ind in enumerate(indices):
            optimizer.zero_grad()
            self.zero_grad()
            data_batch, times_batch = data.get_batch(train_data, ind, args.vocab_size, args.emb_size, temporal=True, times=train_times)
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch

            loss, nll, kl_alpha, kl_eta, kl_theta = self(data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
            optimizer.step()

            acc_loss += torch.sum(loss).item()
            acc_nll += torch.sum(nll).item()
            acc_kl_theta_loss += torch.sum(kl_theta).item()
            acc_kl_eta_loss += torch.sum(kl_eta).item()
            acc_kl_alpha_loss += torch.sum(kl_alpha).item()
            cnt += 1

            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_nll = round(acc_nll / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
                cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
                lr = optimizer.param_groups[0]['lr']
                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))

        cur_loss = round(acc_loss / cnt, 2)
        cur_nll = round(acc_nll / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2)
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2)
        lr = optimizer.param_groups[0]['lr']
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
        print('*'*100)

    def visualize(self, args, vocabulary):
        """Visualizes topics and embeddings and word usage evolution.
        """
        self.eval()
        with torch.no_grad():
            alpha = self.mu_q_alpha
            beta = self.get_beta(alpha)
            print('beta: ', beta.size())
            print('\n')
            print('#'*100)
            print('Visualize topics...')
            times = [0, 10, 40]
            topics_words = []
            for k in range(args.num_topics):
                for t in times:
                    gamma = beta[k, t, :]
                    top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                    topic_words = [vocabulary[a] for a in top_words]
                    topics_words.append(' '.join(topic_words))
                    print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

            print('\n')
            print('Visualize word embeddings ...')
            queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights',  'africa']
            try:
                embeddings = self.rho.weight  # Vocab_size x E
            except:
                embeddings = self.rho         # Vocab_size x E
            neighbors = []
            for word in queries:
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocabulary, args.num_words)))
            print('#'*100)

    def _eta_helper(self, rnn_inp):
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()
        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        etas[0] = self.mu_q_eta(inp_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            etas[t] = self.mu_q_eta(inp_t)
        return etas

    def get_eta(self, source, valid_rnn_inp, test_1_rnn_inp):
        self.eval()
        with torch.no_grad():
            if source == 'val':
                rnn_inp = valid_rnn_inp
                return _eta_helper(rnn_inp)
            else:
                rnn_1_inp = test_1_rnn_inp
                return _eta_helper(rnn_1_inp)

    def get_theta(self, eta, bows):
        self.eval()
        with torch.no_grad():
            inp = torch.cat([bows, eta], dim=1)
            q_theta = self.q_theta(inp)
            mu_theta = self.mu_q_theta(q_theta)
            theta = F.softmax(mu_theta, dim=-1)
            return theta

    def get_completion_ppl(self, source, args, validation_data, test_1_data, test_2_data):
        """Returns document completion perplexity.
        """
        validation_data, validation_times = data.get_time_columns(validation_data)
        self.eval()
        with torch.no_grad():
            alpha = self.mu_q_alpha
            if source == 'val':
                indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
                val_data = validation_data
                times = validation_times
                eta = self.get_eta('val')
                acc_loss = 0
                cnt = 0
                for idx, ind in enumerate(indices):
                    data_batch, times_batch = data.get_batch(val_data, ind, args.vocab_size, args.emb_size, temporal=True, times=times)
                    sums = data_batch.sum(1).unsqueeze(1)
                    if args.bow_norm:
                        normalized_data_batch = data_batch / sums
                    else:
                        normalized_data_batch = data_batch

                    eta_td = eta[times_batch.type('torch.LongTensor')]
                    theta = self.get_theta(eta_td, normalized_data_batch)
                    alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                    beta = self.get_beta(alpha_td).permute(1, 0, 2)
                    loglik = theta.unsqueeze(2) * beta
                    loglik = loglik.sum(1)
                    loglik = torch.log(loglik)
                    nll = -loglik * data_batch
                    nll = nll.sum(-1)
                    loss = nll / sums.squeeze()
                    loss = loss.mean().item()
                    acc_loss += loss
                    cnt += 1
                cur_loss = acc_loss / cnt
                ppl_all = round(math.exp(cur_loss), 1)
                print('*'*100)
                print('{} PPL: {}'.format(source.upper(), ppl_all))
                print('*'*100)
                return ppl_all
            else: 
                indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
                eta_1 = self.get_eta('test')
                acc_loss = 0
                cnt = 0
                indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
                for idx, ind in enumerate(indices):
                    test_1_data, test_1_times = data.get_time_columns(test_1_data)
                    data_batch_1, times_batch_1 = data.get_batch(test_1_data,
                                                                 ind,
                                                                 args.vocab_size,
                                                                 args.emb_size,
                                                                 temporal=True,
                                                                 times=test_1_times)
                    sums_1 = data_batch_1.sum(1).unsqueeze(1)
                    if args.bow_norm:
                        normalized_data_batch_1 = data_batch_1 / sums_1
                    else:
                        normalized_data_batch_1 = data_batch_1

                    eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
                    theta = get_theta(eta_td_1, normalized_data_batch_1)
                    test_2_data, test_2_times = data.get_time_columns(test_1_data)
                    data_batch_2, times_batch_2 = data.get_batch(test_2_data,
                                                                 ind,
                                                                 args.vocab_size,
                                                                 args.emb_size,
                                                                 temporal=True,
                                                                 times=test_2_times)
                    sums_2 = data_batch_2.sum(1).unsqueeze(1)

                    alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
                    beta = self.get_beta(alpha_td).permute(1, 0, 2)
                    loglik = theta.unsqueeze(2) * beta
                    loglik = loglik.sum(1)
                    loglik = torch.log(loglik)
                    nll = -loglik * data_batch_2
                    nll = nll.sum(-1)
                    loss = nll / sums_2.squeeze()
                    loss = loss.mean().item()
                    acc_loss += loss
                    cnt += 1
                cur_loss = acc_loss / cnt
                ppl_dc = round(math.exp(cur_loss), 1)
                print('*'*100)
                print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
                print('*'*100)
                return ppl_dc

    def _diversity_helper(self, args, beta, num_tops):
        list_w = np.zeros((args.num_topics, num_tops))
        for k in range(args.num_topics):
            gamma = beta[k, :]
            top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
            list_w[k, :] = top_words
        list_w = np.reshape(list_w, (-1))
        list_w = list(list_w)
        n_unique = len(np.unique(list_w))
        diversity = n_unique / (args.num_topics * num_tops)
        return diversity

    def get_topic_quality(self, args, train_data):
        """Returns topic coherence and topic diversity.
        """
        self.eval()
        with torch.no_grad():
            alpha = self.mu_q_alpha
            beta = self.get_beta(alpha)
            print('beta: ', beta.size())
            print('\n')
            print('#'*100)
            print('Get topic diversity...')
            num_tops = 25
            TD_all = np.zeros((args.num_times,))
            for tt in range(args.num_times):
                TD_all[tt] = _diversity_helper(beta[:, tt, :], num_tops)
            TD = np.mean(TD_all)
            print('Topic Diversity is: {}'.format(TD))

            print('\n')
            print('Get topic coherence...')
            print('train_tokens: ', train_data[0])
            TC_all = []
            cnt_all = []
            for tt in range(args.num_times):
                tc, cnt = get_topic_coherence(beta[:, tt, :].cpu().numpy(), train_data, vocab)
                TC_all.append(tc)
                cnt_all.append(cnt)
            print('TC_all: ', TC_all)
            TC_all = torch.tensor(TC_all)
            print('TC_all: ', TC_all.size())
            print('\n')
            print('Get topic quality...')
            quality = tc * diversity
            print('Topic Quality is: {}'.format(quality))
            print('#'*100)
