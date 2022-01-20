from torch import nn
from torch.nn import functional as F


__all__ = ["UpLoss"]

class UpLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.mse_criterias = nn.MSELoss()

    def forward(self, x, targer):
        low = self.mse_criterias(F.interpolate(x, mode="bilinear", scale_factor=2), targer)
        up = self.mse_criterias(x, F.interpolate(targer, mode="bilinear", scale_factor=0.5))

        return low, up