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
            #DoubleConv(in_channels=in_channels, out_channels=out_channels),

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
    
    def __init__(self, n_blocks: int = 3, in_channels: int = 3, n_features: int = 64, n_features_scale: int = 2) -> None:
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


class NConv2D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_conv: int, kernel_size: int, padding: int = 1) -> None:
        """sequential of n (Conv2D, ReLU), output size same input

        Args:
            in_channels (int): input channels
            n_conv (int): num conv laers
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (intorstr, optional): padding. Defaults to 1.
        """
        super().__init__()

        self.n_conv_body = nn.Sequential(
            *(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding), 
                nn.ReLU()
                ) for _ in range(n_conv - 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.n_conv_body(x)
        return out


class MDown(nn.Module):

    def __init__(self, in_channels: int, n_conv_seq: list, num_feature_scaler: int = 2, conv_kernel_size: int = 3, pool_kernel_size: int = 2) -> None:
        super().__init__()

        self.down_body = nn.Sequential(
            *(nn.Sequential(
                NConv2D(in_channels=in_channels * num_feature_scaler**i, out_channels=in_channels * num_feature_scaler**(i+1), n_conv=n_conv - 1, kernel_size=conv_kernel_size),
                nn.MaxPool2d(kernel_size=pool_kernel_size)
            ) for i, n_conv in enumerate(n_conv_seq))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_body(x)
        return out


class MUp(nn.Module):

    def __init__(self, in_channels: int, n_up_steps: int, num_features_scale: int, kernel_size: int = 3, padding: int = 1, stride: int = 2, tconv_kernel_size: int = 4) -> None:
        super().__init__()

        self.up_body = nn.Sequential(
            *(nn.Sequential(
                nn.ConvTranspose2d(in_channels=n_channels * num_features_scale, out_channels=n_channels, kernel_size=tconv_kernel_size, padding=padding, stride=stride),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU()
            )for n_channels in [in_channels // num_features_scale**(i+1) for i in range(n_up_steps)])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up_body(x)
        return out


class MEncoder(nn.Module):

    def __init__(self, in_channels: int, num_features: int, n_conv_seq: list, kernel_size: int = 3, padding: str = 1) -> None:
        super().__init__()

        self.encoder_body = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=kernel_size, padding=padding),
            MDown(in_channels=num_features, n_conv_seq=n_conv_seq)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder_body(x)
        return out


class MDecoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_up_steps: int, num_features_scale: int, kernel_size: int = 3, padding: str = 1) -> None:
        super().__init__()

        self.decoder_body = nn.Sequential(
            MUp(in_channels=in_channels, n_up_steps=n_up_steps, num_features_scale=num_features_scale),
            nn.Conv2d(in_channels=in_channels // num_features_scale**n_up_steps, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder_body(x)
        return out


class MAEnet(nn.Module):

    def __init__(self, in_channels: int = 3, n_conv_seq: list = [2, 2, 3], num_features: int = 64, num_features_scaler: int = 2) -> None:
        super().__init__()

        n_up_steps = len(n_conv_seq)

        self.encoder = MEncoder(in_channels=in_channels, num_features=num_features, n_conv_seq=n_conv_seq)
        self.decoder = MDecoder(in_channels=num_features * num_features_scaler**n_up_steps, out_channels=in_channels, n_up_steps=n_up_steps, num_features_scale=num_features_scaler)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_out = self.encoder(x)
        d_out = self.decoder(e_out)
        return d_out

    def get_encoder(self) -> MEncoder:
        return self.encoder
