import torch
import numpy as np

def cal_acc(prediction, gt, train_num, distribute=True):
    if not distribute:
        acc = ((prediction == gt).sum())/prediction.shape[0]
    else:
        prediction = (np.stack(prediction)).reshape(-1)[:train_num]
        gt = (torch.stack(gt).cpu().numpy().reshape(-1))[:train_num]
        acc = ((prediction == gt).sum())/prediction.shape[0]
    return acc