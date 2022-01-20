from .base_ds import BaseRoatateDataset

from torchvision import transforms
from PIL import Image

__all__ = ["AutoencoderDataset"]

class AutoencoderDataset(BaseRoatateDataset):

    def __init__(self, gt_path: str, scale: int = 2, resample = Image.BILINEAR, n_rotation: int = 4, ext: str = ".png", size: tuple = (256, 256), img_mode: str = "RGB") -> None:
        super().__init__(gt_path, n_rotation=n_rotation, ext=ext, size=size, img_mode=img_mode)

        self.resample = resample
        self.scale = scale

    
    def __getitem__(self, index):
        hr_img = super().__getitem__(index)[1]
        #sr_img = hr_img.resize((x // 2 for x in hr_img.size), resample=Image.NEAREST).resize(hr_img.size, resample=Image.NEAREST)

        hr = transforms.ToTensor()(hr_img)
        #sr = transforms.ToTensor()(sr_img)

        return hr, hr.clone()

    def __len__(self):
        return super().__len__()