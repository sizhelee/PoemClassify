import torch
from torch.nn.utils.rnn import pad_sequence

global word2idx, label2idx

def load_data():
    word2idx = {"<unk>": 0}
    label2idx = {}
    idx2word = ["<unk>"]
    idx2label = []

    train_data = []
    with open("data/train.txt", encoding='utf-8') as f:
        for line in f:
            text, author = line.strip().split()
            for c in text:
                if c not in word2idx:
                    word2idx[c] = len(idx2word)
                    idx2word.append(c)
            if author not in label2idx:
                label2idx[author] = len(idx2label)
                idx2label.append(author)
            train_data.append((text, author))

    valid_data = []
    with open("data/valid.txt", encoding='utf-8') as f:
        for line in f:
            text, author = line.strip().split()
            valid_data.append((text, author))

    test_data = []
    with open("data/test.txt", encoding='utf-8') as f:
        for line in f:
            text, author = line.strip().split()
            test_data.append((text, author))

    return word2idx, idx2word, label2idx, idx2label, train_data, valid_data, test_data


def label2onehot(label, word2idx):
    '''
    input: label(tensor) N*1
    output: onehot(tensor) N*len(word2idx)
    '''
    text_length = len(label)
    onehot = torch.zeros(text_length, len(word2idx))
    one = torch.ones_like(onehot)

    onehot.scatter_(dim=1, index=label.reshape(text_length,-1).long(), src=one)
    return onehot


def w2i(x, word2idx):
    if x in word2idx.keys():
        return word2idx[x]
    else:
        return 0


def make_data(text, author, word2idx, label2idx):
    """
    输入
        text: str
        author: str
    输出
        x: LongTensor, shape = (1, text_length) -> (1, text_length, input_size)
        y: LongTensor, shape = (1,)
    """
    x_ls, mask = [], []
    for onetext in text:
        x = label2onehot(torch.tensor(list(map(lambda x: w2i(x, word2idx), onetext))), word2idx)
        x_ls.append(x)
        mask.append(torch.ones_like(x[:,0]))
    x = pad_sequence(x_ls).transpose(0, 1)
    mask = pad_sequence(mask).T
    y = torch.tensor([label2idx[i] for i in author])

    return x, y, mask