import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

from cfgs import train_test_cfg as cfg
from mutils.data import SRganDataset
from mutils.models import Discriminator64, MGenerator

from mutils.losses import EncoderLoss, SSIMLoss, StyleLoss, RescaleLoss, LossSaver
  
CUDA_LAUNCH_BLOCKING=1

#check out directory

if not os.path.exists(cfg.weights_out_path):
    os.makedirs(cfg.weights_out_path)

print(f"name: {cfg.name}")

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
lmu_criterias = RescaleLoss(lambda x, y: mse_criterias(x, y) + enc_criterias(x, y)).to(device)


#init loss writers

loss_data_path = os.path.join(cfg.loss_data_path, f"l_{cfg.name}")
adv_writer = LossSaver("adv", loss_data_path)
dis_writer = LossSaver("dis", loss_data_path)
mse_writer = LossSaver("mse", loss_data_path)



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
weights_out_path = os.path.join(cfg.weights_out_path, f"w_{cfg.name}")
if not os.path.exists(weights_out_path):
        os.mkdir(weights_out_path)

for epoch in range(cfg.start_epoch, cfg.epochs):
    for index, (input, target) in enumerate(dataloader, 1):

        input, target = input.to(device), target.to(device)


        # discriminator train stap

        upx2 = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")

        g_out = generator(input)
        g_out = upx2(g_out)
        d_optimizer.zero_grad()

        d_input = torch.cat((g_out, target)).to(device)
        d_out = torch.sigmoid(discriminator(d_input)).to(device)

        a1, a2 = tuple(d_out.size())
        size = (a1 // 2, a2)
        real_sample = torch.full(size, 1.).to(device)
        fake_sample = torch.full(size, 0.).to(device)
        d_target = torch.cat((fake_sample, real_sample)).to(device)
        dis_loss =  mse_criterias(d_out, d_target)
        
        dis_loss.backward()
        d_optimizer.step()
        dis_writer(dis_loss.item())


        # generator train stap

        g_optimizer.zero_grad()
        g_out = generator(input)
        d_out = torch.sigmoid(discriminator(upx2(g_out)))
        size = d_out.size()
        real_sample = torch.full(size, 1.).to(device)

        adv_loss = 0.001 * mse_criterias(d_out, real_sample)
        #low_loss, med_loss, up_loss = lmu_criterias(g_out, target)
        #ssim_loss = ssim_criterias(g_out, target)
        #g_loss = adv_loss + low_loss + med_loss + up_loss
        
        low_loss, med_loss, up_loss = lmu_criterias(g_out, target)
        
        g_loss = adv_loss + up_loss
        g_loss.backward()
        g_optimizer.step()


        # log info and save state
        
        n_iter = 2000
        if index % n_iter == 0 and False:
            torch.save(generator.state_dict(), os.path.join(weights_out_path, f"g_epoch_{epoch+1}_{index // n_iter}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(weights_out_path, f"d_epoch_{epoch+1}_{index // n_iter}.pth"))

        print(f"epoch: {epoch}/{cfg.epochs}\t| iter: {index}/{nbatches}\t| adv: {adv_loss.item(): .6f}\t| dis: {dis_loss.item():.6f}\t| up: {up_loss.item():.6f}")

    if epoch + 1 in [100, 200]:
        torch.save(generator.state_dict(), os.path.join(weights_out_path, f"g_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(weights_out_path, f"d_epoch_{epoch+1}.pth"))
