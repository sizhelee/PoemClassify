import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from tqdm import tqdm, trange

from model import EncoderRNN
from preprocess import load_data, make_data

device = torch.device("cuda")

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def train(model, batch_size, train_data, loss_fn, optimizer, word2idx, label2idx):
    model.train()
    
    for iter in range(len(train_data)//batch_size):

        samples = train_data[iter*batch_size:(iter+1)*batch_size]

        sentence, label = [sample[0] for sample in samples], [sample[1] for sample in samples]
        x, y, mask = make_data(sentence, label, word2idx, label2idx)

        optimizer.zero_grad()
        y_pred = model(x, mask)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            print("[Train] iter {}/{}, loss:{}".format(iter, len(train_data)//batch_size, loss))


def main():

    word2idx, idx2word, label2idx, idx2label, train_data, valid_data, test_data = load_data()
    epochs = 1
    batch_size = 128

    model = EncoderRNN(len(word2idx), len(word2idx), 512, 5, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, batch_size, train_data, criterion, optimizer, word2idx, label2idx)


if __name__ == "__main__":
    main()