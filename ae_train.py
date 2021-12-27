import os
import torch

from torch import nn
from torch.utils.data import dataloader
from torch import optim

from mutils.losssaver import LossSaver
from mutils.models.aenet import MAEnet
from mutils.data.data_utils import AutoencoderDataset

from mutils.losses.ssim import SSIM


def save_state(path: str, model):
    torch.save({
                "model_state_dict": model.state_dict(),
            }, path)

def train(dloader, model, criterius, optimizer, device, loss_saver, results_path: str, epoch: int):

    n_batches = len(dloader)
    
    for batch, (X, y) in enumerate(dloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = criterius(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch: {epoch} | batch number: {batch}/{n_batches} | loss: {loss.item():.7f}")
        loss_saver(loss.item())

        #if batch % 20000 == 0:
         #   save_state(path=os.path.join(results_path, f"epoch_{epoch}_it_{batch}.pth"), model=model)


def main():

    name = "test"
    nepochs = 500
    lr = 0.0001
    gt_path = r"./../data/tests/fhalo_test"
    batch_size = 4
    patch_size = (32, 32)
    results_path = os.path.join(r"./autoencoder/results/", f"w_{name}")
    pretained_wpath = r"./autoencoder/results/w_test/epoch_99.pth"
    start_epoch = 100

    os.makedirs(results_path, exist_ok=True)

    loss_saver = LossSaver("mse", os.path.join(r"./autoencoder/loss_data/", f"l_{name}"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MAEnet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not pretained_wpath is None:
        state_dict = torch.load(pretained_wpath)
        model.load_state_dict(state_dict["model_state_dict"])

    mse_crirerius = SSIM().to(device)

    dataset = AutoencoderDataset(gt_path=gt_path, size=patch_size, img_mode="YCbCr")
    dloader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"found {len(dataset)} samples")

    for epoch in range(start_epoch, nepochs):
        train(dloader=dloader, model=model, criterius=mse_crirerius, optimizer=optimizer, device=device, loss_saver=loss_saver, epoch=epoch, results_path=results_path)
        save_state(path=os.path.join(results_path, f"epoch_{epoch}.pth"), model=model)
            


if __name__ == "__main__":
    main()
