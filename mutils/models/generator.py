from basicsr.archs.srresnet_arch import MSRResNet
from torch.nn import functional as F

__all__ = ["MGenerator"]

class MGenerator(MSRResNet):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super().__init__(num_in_ch, num_out_ch, num_feat, num_block, upscale)


    def forward(self, x):
        out = super().forward(x)
        out = F.interpolate(out, mode="bilinear", scale_factor=2)
        return out