import torch
import numpy as np
import os
import logging

def cal_acc(prediction, gt, train_num, distribute=True):
    if not distribute:
        acc = ((prediction == gt).sum())/prediction.shape[0]
    else:
        prediction = (np.stack(prediction)).reshape(-1)[:train_num]
        gt = (torch.stack(gt).cpu().numpy().reshape(-1))[:train_num]
        acc = ((prediction == gt).sum())/prediction.shape[0]
    return acc

def init_logger(logging_name=''):
    
    log_path = "train.log"
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def write_log(log, info, verbose=False):
    log.info(info)