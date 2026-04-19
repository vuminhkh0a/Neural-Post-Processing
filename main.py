import torch
from model import *
from data import *
from train import * 
import os
import sys
import argparse



IRMAS_PATH_CP = './checkpoints/IRMAS_checkpoints'
LIBRISPEECH_PATH_CP = './checkpoints/LibriSpeech_checkpoints'

MODEL_NAME = 'UnetAttention'
DATASET_NAME = 'LibriSpeech'
BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY= True
DEVICE = 'cuda'


EPOCHS = 11
LR = 1e-4

def init(model_name, dataset_name, batch_size, num_workers, pin_memory, lr, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(device)
    sys.stdout.flush()

    if model_name == 'Autoencoder':
        model = Autoencoder().to(device)
    elif model_name == 'Unet':
        model = Unet().to(device)
    elif model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(encoder_name="vgg11_bn", encoder_depth=5, activation=None, in_channels=1, classes=1).to(device)
    elif model_name == 'UnetAttention':
        model = AttU_Net(img_ch=1, output_ch=1).to(device)
    elif model_name == 'Proposed':
        model = Proposed(img_ch=1, output_ch=1).to(device)

    train_loader, val_loader, test_loader = get_loaders(dataset_name, batch_size, num_workers, pin_memory)

    if dataset_name == 'IRMAS':
        checkpoint = os.path.join(IRMAS_PATH_CP, dataset_name + '_' + model_name + '.pth')
    elif dataset_name == 'LibriSpeech':
        checkpoint = os.path.join(LIBRISPEECH_PATH_CP, dataset_name + '_' + model_name + '.pth')


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return device, model, train_loader, val_loader, test_loader, checkpoint, optimizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == "true", default=PIN_MEMORY)

    parser.add_argument("--device", type=str, default=DEVICE)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)

    parser.add_argument("--mode", type=str, default='train')

    return parser.parse_args()


def main():
    # args = get_args()

    # device, model, train_loader, val_loader, test_loader, checkpoint, optimizer = init(
    #     model_name=args.model_name,
    #     dataset_name=args.dataset_name,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory,
    #     lr=args.lr,
    #     device=args.device
    # )

    # if args.mode == 'train':
    #     train_one_dataset(EPOCHS, model, optimizer, device, train_loader, val_loader, checkpoint)

    device, model, train_loader, val_loader, test_loader, checkpoint, optimizer = init(model_name=MODEL_NAME, 
                                                                                        dataset_name=DATASET_NAME, 
                                                                                        batch_size=BATCH_SIZE, 
                                                                                        num_workers=NUM_WORKERS, 
                                                                                        pin_memory=PIN_MEMORY,
                                                                                        lr=LR,
                                                                                        device=DEVICE)
    
    train_one_dataset(EPOCHS, model, optimizer, device, train_loader, val_loader, checkpoint)

    

    model.load_state_dict(torch.load(checkpoint))
    mse, snr = evaluate(model, device, test_loader, with_pesq_stoi=False)
    # mse, snr, pseq, stoi = evaluate(model, device, test_loader, with_pesq_stoi=True)


    

    print(f'{DATASET_NAME} - {MODEL_NAME} test results:\n')
    # print(f"MSE: {mse:.4f} | SNR: {snr:.4f} | PSEQ: {pseq:.4f} | STOI: {stoi:.4f}")
    print(f"MSE: {mse:.4f} | SNR: {snr:.4f}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()