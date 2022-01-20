import numpy as np
import os
import torch
import time as time

from basicsr.archs.srresnet_arch import MSRResNet
from PIL import Image
from skimage import io, metrics
from torchvision import transforms

from cfgs import test_cfg1 as cfg




def calc_metrics(sr_path: str, gt_path: str, names: list = None):
    if names == None:
        names = os.listdir(sr_path)

    psnr = []
    ssim = []
    for name in names:
        sr_img = io.imread(os.path.join(sr_path, name))
        hr_img = io.imread(os.path.join(gt_path, name))

        try:
            psnr.append(metrics.peak_signal_noise_ratio(sr_img, hr_img))
            ssim.append(metrics.structural_similarity(  sr_img,
                                                        hr_img,
                                                        win_size=11,
                                                        gaussian_weights=True,
                                                        multichannel=True,
                                                        data_range=255.0,
                                                        K1=0.01,
                                                        K2=0.03,
                                                        sigma=1.5))
        except Exception as ex:
            print(f"indalid saved image {name}")

        if __name__ == "__main__":
            print(f"PSNR: {psnr[-1]:.4f}, SSIM: {ssim[-1]:.4f} | {name}")
    if __name__ == "__main__":
        print(f"MEAN | PSNR: {np.mean(psnr):.4f}, SSIM: {np.mean(ssim):.4f}")
    return np.mean(psnr), np.mean(ssim)





def predict(wights_path: str, out_data_path: str, test_data_path: str, img_mode: str, names: list = None):

    if names == None:
        names = os.listdir(test_data_path)


    #init device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #init and load model

    model = MSRResNet(upscale=2).to(device=device)
    model.load_state_dict(torch.load(wights_path))


    #proces data

    os.makedirs(out_data_path, exist_ok=True)
    for name in names:
        lr_img = Image.open(os.path.join(test_data_path, name)).convert(img_mode)
        input = transforms.ToTensor()(lr_img).view(1, -1, lr_img.size[1], lr_img.size[0]).to(device)
        lr_img.close()
        dt = -time.time()
        output = model(input).cpu()
        
        output = output.detach().apply_(lambda x: 0 if x < 0 else (1 if x > 1 else x))
        if __name__ == "__main__":
            print(f"nan status: {torch.sum(torch.isnan(output))}")
        dt += time.time()
        sr_img = transforms.ToPILImage(mode=img_mode)(output[0]).convert("RGB")
        sr_img.save(os.path.join(out_data_path, name))
        sr_img.close()
        if __name__ == "__main__":
            print(f"{name} saved\t|\ttime: {dt}")


# main

if __name__ == "__main__":

    print("start testing...")
    names = [name for name in os.listdir(cfg.test_data_path) if name.endswith(cfg.ext)]


    #predictaa

    print("predict...")
    predict(cfg.wights_path, cfg.out_data_path, cfg.test_data_path, cfg.img_mode, names)


    #calc metrics

    if cfg.calc_metrics:
        print("calculate metrics")

        print("BICUBIC")
        calc_metrics(cfg.bic_sr_path, cfg.hr_data_path, names)
        print("MODEL")
        calc_metrics(cfg.out_data_path, cfg.hr_data_path, names)