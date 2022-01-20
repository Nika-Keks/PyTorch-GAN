import os
import torch

from basicsr.archs.srresnet_arch import MSRResNet
from torch.utils.data.dataloader import DataLoader

from cfgs import train_test_cfg as cfg
from mutils.data import SRganDataset, NSRganDataset
from mutils.models import Discriminator64, MGenerator

from mutils.losses import EncoderLoss, SSIMLoss, StyleLoss, StyleEncoderLoss, UpLoss, LossSaver
  
CUDA_LAUNCH_BLOCKING=1

#check out directory

if not os.path.exists(cfg.weights_out_path):
    os.makedirs(cfg.weights_out_path)


#init device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#init generator

generator = MGenerator(upscale=cfg.upscale).to(device=device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr[0])


#init discriminator

discriminator = Discriminator64(3, 64).to(device=device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr[1])


#init criteriases

mse_criterias = torch.nn.MSELoss().to(device=device)
#enc_criterias = EncoderLoss(model_state_dict_path=cfg.encoder_state).to(device=device)
ssim_criterias = SSIMLoss(window_size=3)
stl_criterias = StyleLoss()
#sen_criterias = StyleEncoderLoss(model_state_dict_path=cfg.encoder_state, layer=3).to(device=device)
mme_criterias = torch.nn.L1Loss().to(device=device)
lup_criterias = UpLoss().to(device)


#init loss writers

loss_data_path = os.path.join(cfg.loss_data_path, f"l_{cfg.name}")
enc_writer = LossSaver("enc", loss_data_path)
adv_writer = LossSaver("adv", loss_data_path)
dis_writer = LossSaver("dis", loss_data_path)
sen_writer = LossSaver("sen", loss_data_path)
stl_writer = LossSaver("stl", loss_data_path)
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

dataset = NSRganDataset(**cfg.datasetinit.tokwargs(), mean=0., std=0.01)
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

        g_out = generator(input)
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
        d_out = torch.sigmoid(discriminator(g_out))
        size = d_out.size()
        real_sample = torch.full(size, 1.).to(device)

        adv_loss = 0.001 * mse_criterias(d_out, real_sample)
        mse_loss = mse_criterias(g_out, target)

        g_loss = adv_loss + mse_loss
        g_loss.backward()
        g_optimizer.step()

        # adv_writer(adv_loss.item())
        #mse_writer(mse_loss.item())


        # log info and save state
        
        n_iter = 2000
        if index % n_iter == 0:
            torch.save(generator.state_dict(), os.path.join(weights_out_path, f"g_epoch_{epoch+1}_{index // n_iter}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(weights_out_path, f"d_epoch_{epoch+1}_{index // n_iter}.pth"))

        print(f"epoch: {epoch}/{cfg.epochs}\t| iter: {index}/{nbatches}\t| adv: {adv_loss.item(): .6f}\t| dis: {dis_loss.item():.6f}\t| mse: {mse_loss.item():.6f}")
        
    torch.save(generator.state_dict(), os.path.join(weights_out_path, f"g_epoch_{epoch+1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(weights_out_path, f"d_epoch_{epoch+1}.pth"))
