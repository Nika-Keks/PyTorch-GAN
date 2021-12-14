import torch
import os
import sys

from PIL import Image
from torchvision import transforms

from aenet import AEnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(img, model: AEnet, mode: str):
    tensor_lr_img = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0]).to(device=device)
    tensor_hr_img = model(tensor_lr_img).cpu().detach()
    tensor_hr_img.apply_(lambda x: 0 if x < 0 else (1 if x > 1 else x))
    tensor_hr_img = tensor_hr_img[0]
    hr_img = transforms.ToPILImage(mode=mode)(tensor_hr_img)

    print(tensor_hr_img)

    return hr_img


def main():
    test_data_path = os.path.join(sys.path[0], r"../../data/tests/fhalo_test")
    out_path = os.path.join(sys.path[0], r"./../../data/tmp/test1") 
    model_wpath = os.path.join(sys.path[0], r"./results/w_YCbCr_relu_bn_s32/epoch_8.pth")
    mode = "YCbCr"

    model = AEnet().to(device=device)
    model.load_state_dict(torch.load(model_wpath)["model_state_dict"])   

    os.makedirs(out_path, exist_ok=True)

    for name in os.listdir(test_data_path):
        lr_img = Image.open(os.path.join(test_data_path, name)).convert(mode)
        hr_img = predict(img=lr_img, model=model, mode=mode).convert("RGB")
        hr_img.save(os.path.join(out_path, name))
        print(f"saved {name} in {out_path}")



if __name__ == "__main__":
    main()