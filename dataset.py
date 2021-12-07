from warnings import simplefilter
from numpy.core.defchararray import rindex
from numpy.core.fromnumeric import size
import torch
from torch.utils import data
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class CustomDataset(Dataset):
    
    def __init__(self, hr_path: str, scale: int = 4, n_rotation: int = 4, ext: str = ".png"):
        hr_names = os.listdir(hr_path)

        self.names = [name for name in hr_names if name.endswith(ext)]
        self.hr_path = hr_path
        self.scale = scale
        self.n_rotation = n_rotation

        if (len(self.names) == 0):
            ValueError("number of samples is 0")


    def __getitem__(self, index):
        hr_img = Image.open(os.path.join(self.hr_path, self.names[index // self.n_rotation])).convert("YCbCr")
        lr_img = hr_img.resize(size=(hr_img.size[0] // self.scale, hr_img.size[1] // self.scale), resample=Image.LANCZOS)

        angle = 360. / self.n_rotation * (index % self.n_rotation)
        hr_img = hr_img.rotate(angle=angle)
        lr_img = lr_img.rotate(angle=angle)

        lr = transforms.ToTensor()(lr_img)
        hr = transforms.ToTensor()(hr_img)

        return lr, hr


    def __len__(self):
        return len(self.names) * self.n_rotation



class DivideDataset(Dataset):

    def __init__(self, gt_path: str, upscale: int = 4, n_rotation: int = 1, ext: str = ".png", size: tuple = (256, 256), img_mode: str = "RGB") -> None:
        self.gt_path = gt_path
        self.upscale = upscale
        self.ext = ext
        self.size = size
        self.n_rotation = n_rotation
        self.img_mode = img_mode
        self.downsampling = [
            Image.BICUBIC, 
            Image.LANCZOS, 
            Image.NEAREST, 
            Image.BILINEAR
        ]

        self.names = self._readNames()
        self.n_entres = [self._devideon(Image.open(os.path.join(gt_path, name))) for name in self.names]
        for i in range(len(self.n_entres) - 1):
            self.n_entres[i + 1] += self.n_entres[i]

        self.img_buffer = None
        self.last_name_index = None
    

    def __getitem__(self, index):
        ds_index = index % len(self.downsampling)
        index = index // len(self.downsampling)

        hr_img = self._loadImage(index // self.n_rotation).convert(self.img_mode)
        lr_img = hr_img.resize(size=(hr_img.size[0] // self.upscale, hr_img.size[1] // self.upscale), resample=self.downsampling[ds_index])

        angle = 360. / self.n_rotation * (index % self.n_rotation)
        hr_img = hr_img.rotate(angle=angle)
        lr_img = lr_img.rotate(angle=angle)

        lr = transforms.ToTensor()(lr_img)
        hr = transforms.ToTensor()(hr_img)

        return lr, hr

    
    def __len__(self):
        return self.n_entres[-1] * self.n_rotation * len(self.downsampling)
    

    def _devideon(self, img: Image.Image) -> int:
        w = img.size[0] // self.size[0]
        h = img.size[1] // self.size[1]

        if w * h == 0:
            RuntimeError(f"invalid input image")

        return  w * h


    def _indexof(self, index: int):
        for i in range(len(self.n_entres)):
            if self.n_entres[i] > index:
                return i, index - ( 0 if i == 0 else self.n_entres[i - 1]) 

        ValueError(f"index {index} out of bounds. len = {len(self.n_entres)}")


    def _getsubimg(self, subindex: int):
        sx, sy = self.img_buffer.size
        wi = subindex // (sx // self.size[0])
        hi = subindex %  (sy // self.size[1])
        x, y = wi * self.size[0], hi * self.size[1]
        
        return self.img_buffer.crop((x, y,  x + self.size[0], y + self.size[1]))


    def _loadImage(self, index):
        name_index, subindex = self._indexof(index)
        if (self.last_name_index is None) or self.last_name_index != name_index:
            self.img_buffer = Image.open(os.path.join(self.gt_path, self.names[name_index]))
            self.last_name_index = name_index

        return self._getsubimg(subindex)

    def _readNames(self) -> list:
        names = []

        for name in os.listdir(self.gt_path):
            if not name.endswith(self.ext):
                continue
            img_path = os.path.join(self.gt_path, name)
            img = Image.open(img_path)

            valid_size = lambda ix, x: ix // x > 0

            ix, iy = img.size
            x, y = self.size

            if valid_size(ix, x) and valid_size(iy, y):
                names.append(name)

        return names


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