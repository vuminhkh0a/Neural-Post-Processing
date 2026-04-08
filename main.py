import torch
from model import *
from data import *
from train import * 
import os
import sys


IRMAS_PATH_CP = './checkpoints/IRMAS_checkpoints'
LIBRISPEECH_PATH_CP = './checkpoints/LibriSpeech_checkpoints'

MODEL_NAME = 'Unet'
DATASET_NAME = 'IRMAS'
INPUT_TYPE = 'spectrogram'
BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY= True
DEVICE = 'cuda:1'

EPOCHS = 100
LR = 1e-4

def init(model_name, dataset_name, input_type, batch_size, num_workers, pin_memory, lr):
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(device)
    sys.stdout.flush()

    if model_name == 'Autoencoder':
        model = Autoencoder().to(device)
    elif model_name == 'Unet':
        model = Unet().to(device)

    train_loader, val_loader, test_loader = get_loaders(dataset_name, input_type, batch_size, num_workers, pin_memory)

    if dataset_name == 'IRMAS':
        checkpoint = os.path.join(IRMAS_PATH_CP, dataset_name + '_' + model_name + '_' + input_type + '.pth')
    elif dataset_name == 'LibriSpeech':
        checkpoint = os.path.join(LIBRISPEECH_PATH_CP, dataset_name + '_' + model_name + '_' + input_type + '.pth')


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return device, model, train_loader, val_loader, test_loader, checkpoint, optimizer


def main():
    device, model, train_loader, val_loader, test_loader, checkpoint, optimizer = init(model_name=MODEL_NAME, 
                                                                                        dataset_name=DATASET_NAME, 
                                                                                        input_type=INPUT_TYPE, 
                                                                                        batch_size=BATCH_SIZE, 
                                                                                        num_workers=NUM_WORKERS, 
                                                                                        pin_memory=PIN_MEMORY,
                                                                                        lr=LR)
    train_one_dataset(EPOCHS, model, optimizer, device, train_loader, val_loader, checkpoint)

    model.load_state_dict(torch.load(checkpoint))
    mse, snr = evaluate(model, device, test_loader)
    # mse, snr, pseq, stoi = evaluate(model, device, test_loader)


    

    print(f'{DATASET_NAME} - {MODEL_NAME} test results:\n')
    # print(f"MSE: {mse:.4f} | SNR: {snr:.4f} | PSEQ: {pseq:.4f} | STOI: {stoi:.4f}")
    print(f"MSE: {mse:.4f} | SNR: {snr:.4f}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()