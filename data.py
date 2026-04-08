import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
from transforms import *

IRMAS_PATH = './data/IRMAS'
LIBRISPEECH_PATH = './data/LibriSpeech'

class CustomDataset(Dataset):
    def __init__(self, degraded_wav_data_path, wav_data_path, dataset_name, is_train, input_type="spectrogram", n_fft=1022, hop_length=512):
        self.degraded_wav_data_path = degraded_wav_data_path
        self.wav_data_path = wav_data_path
        self.dataset_name = dataset_name
        self.is_train = is_train

        self.input_type = input_type
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.max_length = 1048064
    
    def __len__(self):
        return len(self.degraded_wav_data_path)
    
    def __getitem__(self, i):
        x, sr_x = librosa.load(self.degraded_wav_data_path[i], sr=32000)
        y, sr_y = librosa.load(self.wav_data_path[i], sr=32000)

        x = self.change_length(x)
        y = self.change_length(y)

        if self.input_type == "spectrogram":
            x = self.to_spec(x)
            y = self.to_spec(y)


        elif self.input_type == "waveform":
            x = x
            y = y

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y
    

    def change_length(self, x):
        n = x.shape[0]

        if n > self.max_length:
            x = x[:self.max_length]
        
        elif n < self.max_length:
            pad_size = self.max_length - n
            return np.pad(array=x, pad_width=(0, pad_size), mode='constant', constant_values=0)
        
        return x

    def to_spec(self, x):
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)))
        return spec


def get_paths(dataset_name):
    train_x, val_x, test_x = [], [], []
    train_y, val_y, test_y = [], [], []

    if dataset_name == 'IRMAS':
        dataset_path = IRMAS_PATH
    elif dataset_name == 'LibriSpeech':
        dataset_path = LIBRISPEECH_PATH
        
    DATASET_WAV_PATH = os.path.join(dataset_path, 'wav')
    DATASET_DEGRADED_WAV_PATH = os.path.join(dataset_path, 'degraded_wav')

    for fol_type in os.listdir(DATASET_WAV_PATH):
        fol = os.path.join(DATASET_WAV_PATH, fol_type)

        for file_path in os.listdir(fol):
            wav_file_path = os.path.join(fol, file_path)
            degraded_wav_file_path = os.path.join(DATASET_DEGRADED_WAV_PATH, fol_type, file_path)

            if fol_type == 'train':
                train_x.append(degraded_wav_file_path)
                train_y.append(wav_file_path)

            elif fol_type == 'val':
                val_x.append(degraded_wav_file_path)
                val_y.append(wav_file_path)

            else:
                test_x.append(degraded_wav_file_path)
                test_y.append(wav_file_path)

    return train_x, val_x, test_x, train_y, val_y, test_y


def get_datasets(dataset_name, input_type):

    train_x, val_x, test_x, train_y, val_y, test_y = get_paths(dataset_name)

    train_dataset = CustomDataset(degraded_wav_data_path=train_x, wav_data_path=train_y, dataset_name=dataset_name, input_type=input_type, is_train=True)
    val_dataset = CustomDataset(degraded_wav_data_path=val_x, wav_data_path=val_y, dataset_name=dataset_name, input_type=input_type, is_train=False)
    test_dataset = CustomDataset(degraded_wav_data_path=test_x, wav_data_path=test_y, dataset_name=dataset_name, input_type=input_type, is_train=False)

    return train_dataset, val_dataset, test_dataset


def get_loaders(dataset_name, input_type, batch_size, num_workers=4, pin_memory=True):
    
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_name, input_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    return train_loader, val_loader, test_loader




def spec_to_audio(spec_db, n_fft=1022, hop_length=512):

    if spec_db.ndim == 3 or spec_db.ndim == 4:
        spec_db = spec_db.squeeze()

    spec = librosa.db_to_amplitude(spec_db)

    audio = librosa.griffinlim(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32,
        length=1048064
    )

    return audio