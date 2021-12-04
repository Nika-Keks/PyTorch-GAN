import torch
from torch import nn



class DoubleConv(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int or str = "same", padding_mode: str = "reflect", dilation: int = 1) -> None:
        """
            [Conv -> BN -> ReLU -> Dilated Conv -> BN -> ReLU]
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            # Conv
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            
            # Dilated Conv
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, dilation=dilation),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, pool_kernel_size: int = 2) -> None:
        """
            [Double Conv -> MaxPool]
        """
        super().__init__()

        self.down_block = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_block(x)


class Up(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, tconv_kernel_size: int = 2, dilation: int = 0) -> None:
        """
            [Transpose Conv -> Double Conv]
        """
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=tconv_kernel_size),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, dilation=dilation)
        )
        

    def frward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_block(x)
        

class Encoder(nn.Module):
    """
    """

    def __init__(self, n_blocks: int, in_channels: int, n_feature_scale: int = 2) -> None:
        """
            [Down] * n_blocks
        """
        super().__init__()

        self.body = nn.Sequential(
            *(Down(in_channels=(in_channels // n_feature_scale**i), out_channels=(in_channels // n_feature_scale**(i + 1))) for i in range(n_blocks))
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Decoder(nn.Module):
    """
        [Up] * n_blocks
    """

    def __init__(self, n_blocks: int, in_channels: int, n_features_scale: int) -> None:
        """
        """
        super().__init__()

        self.body = nn.Sequential(
            *(Up(in_channels=in_channels * n_features_scale**i, out_channels=in_channels * n_features_scale**(i + 1)) for i in range(n_blocks))
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AEnet(nn.Module):
    """
    """
    
    def __init__(self, n_blocks: int = 4, in_channels: int = 3, n_features_scale: int = 2) -> None:
        """
            [Encoder -> Decoder]
        """
        super().__init__()

        self.encoder = Encoder(n_blocks=n_blocks, in_channels=in_channels, n_features_scale=n_features_scale)
        self.decoder = Decoder(n_blocks=n_blocks, in_channels=in_channels // (n_features_scale**n_blocks), n_features_scale=n_features_scale)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
