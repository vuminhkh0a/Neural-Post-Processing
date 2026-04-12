import torch
import torch.nn as nn
from pesq import pesq
from pystoi import stoi


# Loss functions and evaluation metrics

MAE_loss = nn.L1Loss(reduction='mean')
MSE_loss = nn.MSELoss(reduction='mean')


def snr_db_metric(original, reconstructed, eps=1e-12):
    noise = original - reconstructed
    p_signal = torch.mean(original ** 2)
    p_noise = torch.mean(noise ** 2) + eps
    return 10 * torch.log10((p_signal + eps) / p_noise)

def pesq_metric(original, reconstructed, sr=16000):
    if original.ndim == 2:
        res = 0.0
        for b in range(original.shape[0]):
            res += pesq(sr, original[b], reconstructed[b], 'wb') 
        res /= original.shape[0]
        return res

    return pesq(sr, original, reconstructed, 'wb') 

def stoi_metric(original, reconstructed, sr=16000):

    if original.ndim == 2:
        res = 0.0
        for b in range(original.shape[0]):
            res += stoi(original[b], reconstructed[b], sr, extended=False) 
        res /= original.shape[0]
        return res

    return stoi(original, reconstructed, sr, extended=False)


