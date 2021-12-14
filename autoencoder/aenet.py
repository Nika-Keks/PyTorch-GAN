import torch
from torch import nn



class DoubleConv(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int or str = 1, padding_mode: str = "reflect", dilation: int = 1) -> None:
        """
            [Conv -> BN -> ReLU -> Dilated Conv -> BN -> ReLU]
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            # Conv
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(), # removed inplase=True
            
            # Dilated Conv
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding * dilation, padding_mode=padding_mode, dilation=dilation),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, pool_kernel_size: int = 2, pool_stride: int = 2) -> None:
        """
            [Double Conv -> MaxPool]
        """
        super().__init__()

        self.down_block = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_block(x)
        return out


class Up(nn.Module):
    """
    """

    def __init__(self, in_channels: int, out_channels: int, tconv_kernel_size: int = 4, padding: int = 1, stride: int = 2) -> None:
        """
            [Transpose Conv -> Double Conv]
        """
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=tconv_kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            DoubleConv(in_channels=out_channels, out_channels=out_channels)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up_block(x)
        return out
        

class Encoder(nn.Module):
    """
    """

    def __init__(self, n_blocks: int, in_channels: int, n_features_scale: int, n_features: int) -> None:
        """
            [Down] * n_blocks
        """
        super().__init__()

        self.body = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=n_features),
            *(Down(in_channels=(n_features * n_features_scale**i), out_channels=(n_features * n_features_scale**(i + 1))) for i in range(n_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Decoder(nn.Module):
    """
        [Up] * n_blocks
    """

    def __init__(self, n_blocks: int, in_channels: int, out_channels: int, n_features_scale: int) -> None:
        """
        """
        super().__init__()

        self.body = nn.Sequential(
            *(Up(in_channels=in_channels // n_features_scale**i, out_channels=in_channels // n_features_scale**(i + 1)) for i in range(n_blocks)),
            DoubleConv(in_channels=in_channels // n_features_scale**n_blocks, out_channels=out_channels)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class AEnet(nn.Module):
    """
    """
    
    def __init__(self, n_blocks: int = 4, in_channels: int = 3, n_features: int = 32, n_features_scale: int = 2) -> None:
        """
            [Encoder -> Decoder]
        """
        super().__init__()

        self.encoder = Encoder(n_blocks=n_blocks, in_channels=in_channels, n_features=n_features, n_features_scale=n_features_scale)
        self.decoder = Decoder(n_blocks=n_blocks, in_channels=n_features * (n_features_scale**n_blocks), out_channels=in_channels, n_features_scale=n_features_scale)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encode_x = self.encoder(x)
        out = self.decoder(encode_x)
        return out


    def get_encoder(self) -> Encoder:
        """Return encoder

        Returns:
            Encoder: - 
        """
        return self.encoder
