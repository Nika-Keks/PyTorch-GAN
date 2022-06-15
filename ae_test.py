import torch
import os
import sys

from PIL import Image
from torchvision import transforms

from mutils.models.aenet import MAEnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(img, model, mode: str, res_mode: bool):
    tensor_lr_img = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0]).to(device=device)
    tensor_hr_img = model(tensor_lr_img).cpu().detach()
    tensor_hr_img.apply_(lambda x: 0 if x < 0 else (1 if x > 1 else x))
    tensor_hr_img = tensor_hr_img[0]
    hr_img = transforms.ToPILImage(mode=mode)(tensor_hr_img)

    #print(tensor_hr_img)

    return prepare_res_image(img, hr_img, mode=mode) if res_mode else hr_img

def prepare_res_image(hr_img: Image.Image, res_img: Image.Image, mode: str):
    sr_img = hr_img.resize((x // 2 for x in hr_img.size), resample=Image.NEAREST).resize(hr_img.size, resample=Image.NEAREST)
    out = transforms.ToTensor()(sr_img) + transforms.ToTensor()(res_img)
    out.apply_(lambda x: 0 if x < 0 else (1 if x > 1 else x))
    out_img = transforms.ToPILImage(mode=mode)(out)

    return out_img 



def main():
    test_data_path = os.path.join(sys.path[0], r".\..\data\tests\fhalo_test")
    out_path = os.path.join(sys.path[0], r".\..\data\tmp\test1") 
    model_wpath = os.path.join(sys.path[0], r".\autoenc\results\w_chalo_mse_s_64x64_m_ycbcr\epoch_7.pth")
    mode = "YCbCr"
    res_mode = False

    model = MAEnet().to(device=device)
    model.load_state_dict(torch.load(model_wpath)["model_state_dict"])   

    os.makedirs(out_path, exist_ok=True)

    for name in os.listdir(test_data_path):
        lr_img = Image.open(os.path.join(test_data_path, name)).convert(mode)
        hr_img = predict(img=lr_img, model=model, mode=mode, res_mode=res_mode).convert("RGB")
        hr_img.save(os.path.join(out_path, name))
        print(f"saved {name} in {out_path}")



if __name__ == "__main__":
    main()