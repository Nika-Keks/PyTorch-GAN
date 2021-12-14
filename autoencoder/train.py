import os
import torch

from torch import nn
from torch.utils.data import dataloader
from torch import optim

from losssaver import LossSaver
from aenet import AEnet
from data_utils import AutoencoderDataset


def train(dloader, model, criterius, optimizer, device, loss_saver, epoch: int):

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


def main():

    nepochs = 20
    start_epoch = 0
    lr = 0.001
    gt_path = r"./../data/fhalo"
    batch_size = 64
    patch_size = (32, 32)
    results_path = r"./autoencoder/results/w_YCbCr_relu_bn_s32"
    pretained_wpath = None #r"./autoencoder/results/w_YCbCr_nobn/epoch_1.pth"

    os.makedirs(results_path, exist_ok=True)

    loss_saver = LossSaver("mse", r"./autoencoder/loss_data/losses")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AEnet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not pretained_wpath is None:
        state_dict = torch.load(pretained_wpath)
        model.load_state_dict(state_dict["model_state_dict"])

    mse_crirerius = nn.MSELoss().to(device)

    dataset = AutoencoderDataset(gt_path=gt_path, size=patch_size, img_mode="YCbCr")
    dloader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"found {len(dataset)} samples")

    for epoch in range(start_epoch, nepochs):
        train(dloader=dloader, model=model, criterius=mse_crirerius, optimizer=optimizer, device=device, loss_saver=loss_saver, epoch=epoch)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
        }, os.path.join(results_path, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()