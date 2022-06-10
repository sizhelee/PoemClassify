import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from tqdm import tqdm, trange

from model import EncoderRNN
from preprocess import load_data, make_data
from utils import cal_acc, init_logger, write_log

device = torch.device("cuda")

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def train(epoch, model, batch_size, train_data, loss_fn, optimizer, word2idx, label2idx):

    model.train()
    model.cuda()
    
    for iter in range(len(train_data)//batch_size):

        samples = train_data[iter*batch_size:(iter+1)*batch_size]

        sentence, label = [sample[0] for sample in samples], [sample[1] for sample in samples]
        x, y, mask = make_data(sentence, label, word2idx, label2idx)
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()

        optimizer.zero_grad()
        outputs = model(x, "Train", mask)

        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        batch_acc = cal_acc(y_pred, y.cpu().numpy(), None, False)

        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        if iter % 10 == 0:
            write_log("[Train]epoch {}, iter {}/{}, accuracy: {}, loss:{}".format(
                epoch, iter, len(train_data)//batch_size, round(batch_acc, 5), round(loss.item(), 5)))


def val(model, data, word2idx, label2idx, mode="Valid"):
    
    model.eval()
    model.cpu()

    sentence, label = [sample[0] for sample in data], [sample[1] for sample in data]
    x, y, mask = make_data(sentence, label, word2idx, label2idx)
    outputs = model(x, mode, mask)
    y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    val_acc = cal_acc(y_pred, y.cpu().numpy(), None, False)

    write_log("[{}]accuracy: {}".format(mode, round(val_acc, 5)))
    return val_acc
    


def main():

    word2idx, idx2word, label2idx, idx2label, train_data, valid_data, test_data = load_data()
    epochs = 5
    batch_size = 128

    model = EncoderRNN(len(word2idx), len(word2idx), 512, 5, batch_size)
    # model = torch.load("best.pkl")
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    max_acc = val(model, valid_data, word2idx, label2idx, mode="Valid")
    # max_acc = 0
    for epoch in range(epochs):
        random.shuffle(train_data)
        print("Training epoch {}...".format(epoch+1))
        train(epoch, model, batch_size, train_data, criterion, optimizer, word2idx, label2idx)
        val_acc = val(model, valid_data, word2idx, label2idx, mode="Valid")
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model, "best.pkl")
            print("Successfully saved model!")

    model = torch.load("best.pkl")
    test_acc = val(model, test_data, word2idx, label2idx, mode="Test")


if __name__ == "__main__":
    train_log = init_logger("train")
    main()