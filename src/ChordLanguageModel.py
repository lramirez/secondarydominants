from random import choice, random, shuffle, randint
import sys
from os import path
from ray import tune
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ChordLanguageModel:
    def __init__(self, data_path, config, model_path=None):
        self.path = data_path
        self.config = config
        self.model_path = model_path
        self.make_dataset()
        self.split(0.8)
        if self.model_path:
            if path.exists(model_path):
                self.model = CLM(self.config['EMBED_DIM'], self.config['HIDDEN_DIM'], self.config['N_LAYERS'], self.vocab_size)
                self.model.load_state_dict(torch.load(self.model_path))
            else:
                print("Training model, please have some coffee while waiting...")
                self.train_model(self.config)
                torch.save(self.model.state_dict(), self.model_path)
        else:
            print("Training model, please have some coffee while waiting...")
            self.train_model(self.config)

    def make_dataset(self):
        df = pd.read_csv(self.path)
        df.chords = df.chords.apply(lambda x: eval(x))
        data_ = df.chords.to_numpy().tolist()
        data = []
        for p in data_:
            if len(p) > 1:
                data.append([c[1] for c in p]+["##END##"])
        #print(data)
        dsize = len(data)
        unique = set([c for p in data for c in p])
        self.chords = list(unique)
        self.vocab = {"##PAD##": 0, "##END##": 1}
        for c in self.chords:
            self.vocab[c] = int(len(self.vocab))
        self.vocab_size = len(self.vocab)
        data = [[self.vocab[c] for c in l] for l in data]
        n2c = {v: k for k, v in self.vocab.items()}
        self.n2c = n2c
        data_with_labels = []
        for line in data:
            data_with_labels.append([line, line[1:]])
        padded = []
        m = 0
        for t in data:
            l = len(t)
            if l > m:
                m = l
        padded = []
        for t in data_with_labels:
            diff = m - len(t[0])
            padded_x = t[0] + [0 for i in range(diff)]
            diff = m - len(t[1])
            padded_y = t[1] + [0 for i in range(diff)]
            d = (np.array(padded_x), np.array(padded_y), len(t[0]))
            padded.append(d)
        self.data = padded
    def split(self, train_size):
        shuffle(self.data)
        train_size = int(train_size*len(self.data))
        self.train = self.data[:train_size]
        test = self.data[train_size:]
        test_x, test_y, test_lengths = zip(*test)
        self.test_x = torch.LongTensor(np.stack(test_x))
        self.test_y = torch.LongTensor(np.stack(test_y))
        self.test_lengths = list(test_lengths)

    def get_minibatch(self, batch_size, n):
        mb = self.train[n*batch_size: min((n+1)*batch_size, len(self.train))]
        x, y, lengths = zip(*mb)
        x = np.stack(x)
        y = np.stack(y)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        lengths = list(lengths)
        return x, y, lengths
    def pack_eval_data(self, data):
        m = 0
        data = [[self.vocab[c] for c in p] + [self.vocab['##END##']] for p in data]
        for t in data:
            l = len(t)
            if l > m:
                m = l
        x = []
        lengths = []
        for t in data:
            diff = m - len(t)
            padded_x = t + [0 for i in range(diff)]
            x.append(np.array(padded_x))
            lengths.append(len(t))
        x = torch.LongTensor(np.stack(x))
        lengths = list(lengths)
        return x, lengths
    def get_model(self):
        return self.model
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def evaluate(self, x):
        x, l = self.pack_eval_data(x)
        self.model.eval()
        with torch.no_grad():
            o = self.model(x, l)
        zeros = torch.zeros(o.size(0), dtype=torch.int64).view(-1,1)
        idx = x[:,1:]
        idx = torch.cat((idx, zeros), 1)
        idx = idx.view(o.size(0), o.size(1), -1)
        probs = torch.gather(o, -1, idx).view(o.size(0),-1)
        return probs.sum(dim=1).numpy().tolist()

    def hyperparameter_search(self, config):
        analysis = tune.run(self._hp_search, config=config, verbose=1)
    def _hp_search(self, config):
        self.train_model(config=config, opt=True)

    def train_model(self, config, opt=False):
        HIDDEN_DIM=config['HIDDEN_DIM']
        EMBED_DIM=config['EMBED_DIM']
        L2=config['L2']
        DROPOUT=config['DROPOUT']
        LEARNING_RATE=config['LEARNING_RATE']
        BATCH_SIZE = config['BATCH_SIZE']
        N_LAYERS = config['N_LAYERS']
        N_EPOCHS = config['N_EPOCHS']
        N_BATCHES=int(len(self.train)/BATCH_SIZE)
        model=CLM(EMBED_DIM, HIDDEN_DIM,
                         N_LAYERS, self.vocab_size, dropout=DROPOUT)
        optimizer=optim.Adam(model.parameters(),
                         lr=LEARNING_RATE, weight_decay=L2)
        loss_function=nn.NLLLoss(ignore_index=self.vocab["##PAD##"])
        for epoch in range(N_EPOCHS):
            total_loss=0
            shuffle(self.train)
            model.train()
            for BATCH in range(N_BATCHES):
                x, y, lengths = self.get_minibatch(BATCH_SIZE, BATCH)
                optimizer.zero_grad()
                o=model(x, lengths=lengths)
                o = o.view(-1,o.size(2))
                y = y.view(-1)
                loss=loss_function(o, y)
                loss.backward()
                optimizer.step()
                total_loss += loss
            print("EPOCH "+str(epoch)+": "+str(total_loss.item()))
            model.eval()
            with torch.no_grad():
                test_o=model(self.test_x, lengths=self.test_lengths)
                test_o = test_o.view(-1,test_o.size(2))
                test_y = self.test_y.view(-1)
                test_loss=loss_function(test_o, test_y)
                print("TEST LOSS: "+str(test_loss.item()))
                if opt:
                    tune.report(mean_loss=test_loss)

        print("DONE")
        self.model = model

class CLM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, vocab_size, dropout = 0):
        super(CLM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.LSTM = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first = True)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, lengths=None, batch=True, state=None):
        x_ = self.emb(x)
        l=0
        if batch:
            l = x.size(1)
            x_ = nn.utils.rnn.pack_padded_sequence(x_, lengths, batch_first=True, enforce_sorted=False)
            o, _ = self.LSTM(x_)
        else:
            if x.size(0) == 1:
                o, _ = self.LSTM(x_)
            else:
                o, _ = self.LSTM(x_, state)
        if batch:
            o = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length = l)[0]
        o = self.out(o)
        if batch:
            return F.log_softmax(o, dim=2)
        else:
            return F.log_softmax(o, dim=2), _

