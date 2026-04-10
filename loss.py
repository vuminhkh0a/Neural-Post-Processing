import torch
import torch.nn as nn
from pesq import pesq
from pystoi import stoi


# Loss functions and evaluation metrics

MSE_loss = nn.MSELoss(reduction='mean')
MAE_loss = nn.L1Loss(reduction='mean')

def snr_db_metric(original, reconstructed, eps=1e-12):
    noise = original - reconstructed
    p_signal = torch.mean(original ** 2)
    p_noise = torch.mean(noise ** 2) + eps
    return 10 * torch.log10((p_signal + eps) / p_noise)

def pesq_metric(original, reconstructed, sr=16000):
    return pesq(sr, original.cpu().numpy(), reconstructed.cpu().numpy(), 'wb') 

def stoi_metric(original, reconstructed, sr=16000):
    return stoi(original.cpu().numpy(), reconstructed.cpu().numpy(), sr, extended=False)