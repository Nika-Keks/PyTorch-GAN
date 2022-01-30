from matplotlib.pyplot import sca
from sklearn.preprocessing import scale
from torch import nn
from torch.nn import functional as F


__all__ = ["RescaleLoss"]

class RescaleLoss(nn.Module):

    def __init__(self, criterias) -> None:
        super().__init__()

        self.criterias = criterias

    def forward(self, x, targer):
        low = self.criterias(F.interpolate(x, mode="bilinear", scale_factor=0.5), F.interpolate(targer, mode="bilinear", scale_factor=0.25))
        med = self.criterias(F.interpolate(x, mode="bilinear", scale_factor=2), targer)
        up = self.criterias(x, F.interpolate(targer, mode="bilinear", scale_factor=0.5))

        return low, med, up