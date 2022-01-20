import torch
import numpy as np

from torch import nn

__all__ = ["StyleLoss"]

class StyleLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.criterion(self._gram_mat(x), self._gram_mat(target))

    def _gram_mat(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        features = x.view(c * b, h * w)
        Gmat = torch.mm(features, features.t())

        return Gmat.div(np.sqrt(h * w))