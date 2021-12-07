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

    nepochs = 10
    lr = 0.001
    gt_path = r"./../data/fhalo"
    batch_size = 16
    patch_size = (64, 64)
    results_path = r"./results/w_mYCbCr"

    loss_saver = LossSaver("mse")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AEnet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    mse_crirerius = nn.MSELoss().to(device)

    dataset = AutoencoderDataset(gt_path=gt_path, size=patch_size, img_mode="YCbCr")
    dloader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"found {len(dataset)} samples")

    for epoch in range(nepochs):
        train(dloader=dloader, model=model, criterius=mse_crirerius, optimizer=optimizer, device=device, loss_saver=loss_saver, epoch=epoch)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
        }, os.path.join(results_path, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()