import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet
from torch.utils.data.dataloader import DataLoader

from cfgs import train_halo_cfg as cfg
from dataset import SRganDataset
from discriminator import Discriminator64

from autoencoder.losssaver import LossSaver
from autoencoder.encoder_loss import EncoderLoss


CUDA_LAUNCH_BLOCKING=1

#check out directory

if not os.path.exists(cfg.weights_out_path):
    os.makedirs(cfg.weights_out_path)


#init device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#init generator

generator = MSRResNet(upscale=cfg.upscale).to(device=device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr[0])


#init discriminator

discriminator = Discriminator64(3, 64).to(device=device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr[1])


#init criteriases

mse_criterias = torch.nn.MSELoss().to(device=device)
enc_criterias = EncoderLoss(model_state_dict_path=cfg.encoder_state).to(device=device)


#init loss writers

enc_writer = LossSaver("enc")
adv_writer = LossSaver("adv")
dis_writer = LossSaver("dis")
mse_writer = LossSaver("mse")


#load weighest

if cfg.pretrained:
    if not cfg.weights_path[0] is None:
        generator.load_state_dict(torch.load(cfg.weights_path[0]))
        print(f"load G from {cfg.weights_path[0]}")
    if not cfg.weights_path[1] is None:
        discriminator.load_state_dict(torch.load(cfg.weights_path[1]))
        print(f"load D from {cfg.weights_path[1]}")


#init dataloader

dataset = SRganDataset(**cfg.datasetinit.tokwargs())
dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)
print(f"founded {len(dataset)} samples")

#train loop
nbatches = len(dataloader)
for epoch in range(cfg.start_epoch, cfg.epochs):
    for index, (input, target) in enumerate(dataloader, 1):

        input, target = input.to(device), target.to(device)


        # discriminator train stap

        g_out = generator(input)
        d_optimizer.zero_grad()

        d_input = torch.cat((g_out, target)).to(device)
        d_out = torch.sigmoid(discriminator(d_input)).to(device)

        a1, a2 = tuple(d_out.size())
        size = (a1 // 2, a2)
        real_sample = torch.full(size, 1.).to(device)
        fake_sample = torch.full(size, 0.).to(device)
        d_target = torch.cat((fake_sample, real_sample)).to(device)
        dis_loss = mse_criterias(d_out, d_target)
        
        dis_loss.backward()
        d_optimizer.step()
        dis_writer(dis_loss.item())


        # generator train stap

        g_optimizer.zero_grad()
        g_out = generator(input)
        d_out = torch.sigmoid(discriminator(g_out))
        size = d_out.size()
        real_sample = torch.full(size, 1.).to(device)

        adv_loss = 0.001 * mse_criterias(d_out, real_sample)
        enc_loss = enc_criterias(g_out, target)
        mse_loss = mse_criterias(g_out, target)

        g_loss = adv_loss + mse_loss + enc_loss
        g_loss.backward()
        g_optimizer.step()

        adv_writer(adv_loss.item())
        mse_writer(mse_loss.item())
        enc_writer(enc_loss.item())


        # log info and save state
        
        if index % 5000 == 0:
            torch.save(generator.state_dict(), os.path.join(cfg.weights_out_path, f"g_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(cfg.weights_out_path, f"d_epoch_{epoch+1}.pth"))

        print(f"epoch: {epoch}/{cfg.epochs} | iter: {index}/{nbatches} | adv: {adv_loss.item():.6f} | enc: {enc_loss.item(): .16f} | mse: {mse_loss.item(): .6f} | dis: {dis_loss.item(): .12f}")

    if not os.path.exists(cfg.weights_out_path):
        os.mkdir(cfg.weights_out_path)
        
    torch.save(generator.state_dict(), os.path.join(cfg.weights_out_path, f"g_epoch_{epoch+1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(cfg.weights_out_path, f"d_epoch_{epoch+1}.pth"))
