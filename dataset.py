import os

from torchvision import transforms
from PIL import Image

from autoencoder.data_utils import BaseRoatateDataset


class SRganDataset(BaseRoatateDataset):

    def __init__(self, gt_path: str, n_rotation: int = 4, ext: str = ".png", size: tuple = ..., img_mode: str = "RGB", resampling_mode = Image.BICUBIC, upscale: int = 2) -> None:
        super().__init__(gt_path, n_rotation=n_rotation, ext=ext, size=size, img_mode=img_mode)

        self.resampling_mode = resampling_mode
        self.upscale = upscale

    def __getitem__(self, index):

        hr_img = super().__getitem__(index)[1]
        lr_img = hr_img.resize(size=(x // self.upscale for x in hr_img.size), resample=self.resampling_mode)

        lr = transforms.ToTensor()(lr_img)
        hr = transforms.ToTensor()(hr_img)

        return lr, hr

    def __len__(self):
        return super().__len__()


def mkdata(data_path: str, upscale: int = 2):
    gt_path = os.path.join(data_path, "gt")
    hr_path = os.path.join(data_path, "bicubic")
    lr_path = os.path.join(data_path, f"bic_x{upscale}")
    os.makedirs(hr_path, exist_ok=False)
    os.makedirs(lr_path, exist_ok=False)

    for name in os.listdir(gt_path):
        img = Image.open(os.path.join(gt_path, name))

        source_size = img.size
        img = img.resize((x // upscale for x in source_size), Image.BICUBIC)
        img.save(os.path.join(lr_path, name))
        img = img.resize(source_size, Image.BICUBIC)
        img.save(os.path.join(hr_path, name))
        print(f"prec {name}")


if __name__ == "__main__":
    mkdata(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\halo_percpt")