import torch
import numpy as np
import torchaudio

transform_frequency_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
transform_time_masking = torchaudio.transforms.TimeMasking(time_mask_param=80)

def compose_transform(x):
    '''Apply frequency and time masking to the input tensor x.'''
    return transform_frequency_masking(transform_time_masking(x))