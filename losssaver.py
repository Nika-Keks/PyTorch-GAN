from genericpath import exists
import os

from PIL.Image import Image


class LossSaver():


    def __init__(self, name: str, path: str = os.path.join("loss_data","losses")) -> None:
        file_path = os.path.join(path, f"{name}.loss")
        os.makedirs(path, exist_ok=True)
        self.file = open(file_path, "a")
        


    def __call__(self, x):
        self.file.write(f"{x:.32f}")
        self.file.write("\n")



# import os
# from PIL import Image

# gt_path = r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\bic_x2"
# lr_path = r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\bicubic"
# scale = 2

# for name in os.listdir(gt_path):
#     gt_img = Image.open(os.path.join(gt_path, name))
#     lr_img = gt_img.resize((i * scale for i in gt_img.size), Image.BICUBIC)
#     lr_img.save(os.path.join(lr_path, name))
#     print(f"saved {os.path.join(lr_path, name)}")

# import torch
# from PIL import Image
# from torchvision import transforms
# import numpy as np

# imgt = torch.tensor([[[(i + 1) * (j + 1) / (4*4) for i in range(4)] for j in range(4)] for k in range(3)])

# print(imgt)
# print(torch.sum(torch.isnan(imgt)))
# img = transforms.ToPILImage(mode="RGB")(imgt)
# img.save(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\test_models\ganresnet\out_img\img.png")