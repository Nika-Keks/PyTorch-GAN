import os
import torch

from PIL import Image

__all__ = ["BaseDivideDataset", "BaseRoatateDataset"]

class BaseDivideDataset(torch.utils.data.Dataset):

    def __init__(self, gt_path: str, ext: str = ".png", size: tuple = (256, 256), img_mode: str = "RGB") -> None:
        super().__init__()

        self.gt_path = gt_path
        self.ext = ext
        self.size = size
        self.img_mode = img_mode

        self.names = self._readNames()
        self.n_entres = [self._devideon(Image.open(os.path.join(gt_path, name))) for name in self.names]
        for i in range(len(self.n_entres) - 1):
            self.n_entres[i + 1] += self.n_entres[i]

        self.img_buffer = None
        self.last_name_index = None
    

    def __getitem__(self, index):

        img = self._loadImage(index).convert(self.img_mode)

        return None, img

    
    def __len__(self):
        return self.n_entres[-1]
    

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

class BaseRoatateDataset(BaseDivideDataset):

    def __init__(self, gt_path: str, n_rotation: int = 4, ext: str = ".png", size: tuple = (256, 256), img_mode: str = "RGB") -> None:
        """

        Args:
            gt_path (str): path to ground truth
            n_rotation (int, optional): rotations number. Defaults to 4.
            ext (str, optional): . Defaults to ".png".
            size (tuple, optional): . Defaults to (256, 256).
            img_mode (str, optional): . Defaults to "RGB".
        """
        super().__init__(gt_path, ext=ext, size=size, img_mode=img_mode)

        self.n_rotations = n_rotation

    
    def __getitem__(self, index):
        img = super().__getitem__(index // self.n_rotations)[1]

        angle = 360. / self.n_rotations * (index % self.n_rotations)

        img = img.rotate(angle=angle)

        return None, img

    def __len__(self):
        return super().__len__() * self.n_rotations