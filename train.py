import torch
from loss import *
import sys
from tqdm import tqdm


def train_one_epoch(model, optimizer, device, loader):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        loss = MAE_loss(output, y) + 0.1 * MSE_loss(output, y) 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(loader)



def train_one_dataset(epochs, model, optimizer, device, train_loader, val_loader, checkpoint):
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, optimizer, device, train_loader)
        # mse, snr, pseq, stoi = evaluate(model, device, val_loader)

        # if mse < best_loss:
        #     best_loss = mse
        #     torch.save(model.state_dict(), checkpoint)
        #     print(f" Best model is updated (val_loss={mse:.4f})")

        # print(f"Epochs: [{epoch}/{epochs}] |MSE: {mse:.4f} | SNR: {snr:.4f} | PSEQ: {pseq:.4f} | STOI: {stoi:.4f}")
        # sys.stdout.flush()

        mse, snr = evaluate(model, device, val_loader)

        if mse < best_loss:
            best_loss = mse
            torch.save(model.state_dict(), checkpoint)
            print(f" Best model is updated (val_loss={mse:.4f})")

        print(f"Epochs: [{epoch}/{epochs}] |MSE: {mse:.4f} | SNR: {snr:.4f}")
        sys.stdout.flush()


def evaluate(model, device, loader):
    model.eval()

    running_mse_loss = 0.0
    running_snr_db_metric = 0.0
    running_pseq_metric = 0.0
    running_stoi_metric = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            running_mse_loss += MSE_loss(output, y).item()
            running_snr_db_metric += snr_db_metric(y, output).item()
            # running_pseq_metric += pesq_metric(y_wav, output).item()
            # running_stoi_metric += stoi_metric(y_wav, output).item()


        mse = running_mse_loss / len(loader)
        snr = running_snr_db_metric / len(loader)
        # pseq = running_pseq_metric / len(loader)
        # stoi = running_stoi_metric / len(loader)

    # return mse, snr, pseq, stoi
    return mse, snr



