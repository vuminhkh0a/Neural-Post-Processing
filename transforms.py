import torch
import numpy as np

def min_max_scaler(x):
    max_db = torch.max(x)
    min_db = torch.min(x)
    x = (x - min_db) / (max_db - min_db)
    x = np.clip(x, 0.0, 1.0)
    return x, max_db, min_db

def inverse_min_max_scaler(x, max_db, min_db):
    return x * (max_db - min_db) + min_db