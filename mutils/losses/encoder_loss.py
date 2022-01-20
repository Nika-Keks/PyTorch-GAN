import torch

from torch import nn

from mutils.models.aenet import AEnet, MAEnet

from .style_loss import StyleLoss

__all__ = ["EncoderLoss", "StyleEncoderLoss"]

ENCODER_BODY_NAME = "encoder_body"
DOWN_BODY_NAME = "down_body"

class EncoderLoss(nn.Module):

    def __init__(self, model_state_dict_path: str, criterias_mode: str = "l2", model: str = "maenet") -> None:
        """Loss based on encoder represantation

        Args:
            model_state_dict_path (str): path to train wall model state dict
            criterias_mode (str, optional): criterias for representation, ["l1", "l2"]. Defaults to "l2".

        Raises:
            NotImplementedError: if criterias mode not in ["l1", "l2"]
        """
        super().__init__()

        if model not in ["aenet", "maenet"]:
            raise NotImplementedError(f"unknown model {model}")

        if model == "aenet": 
            autoenc = AEnet()
        else:
            autoenc = MAEnet()
            
        autoenc.load_state_dict(torch.load(model_state_dict_path)["model_state_dict"])
        self.encoder = autoenc.get_encoder()

        if criterias_mode == "l2":
            self.criterias = nn.MSELoss()
        elif criterias_mode == "l1":
            self.criterias = nn.L1Loss()
        else:
            raise NotImplementedError("invalid criterias mode value")



    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x_repres = self.encoder(x)
        y_repres = self.encoder(y)

        loss = self.criterias(x_repres, y_repres)
            
        return loss

class StyleEncoderLoss(EncoderLoss):

    def __init__(self, model_state_dict_path: str, layer: int, criterias_mode: str = "l2", model: str = "maenet") -> None:
        """Style encoder loss from /layer

        Args:
            layer (int): numder of layer
            for the rest see the EncodeLoss

        Raises:
            NotImplementedError: [description]
        """
        super().__init__(model_state_dict_path, criterias_mode=criterias_mode, model=model)

        if model != "maenet":
            raise NotImplementedError(f"for StyleEncoderLoss model mast be maenet, but passed {model}")

        self.style_loss = StyleLoss()

        self.slicer_layer = nn.Sequential(*[
            self.encoder.get_submodule(f"{ENCODER_BODY_NAME}.0"), 
            *[self.encoder.get_submodule(f"{ENCODER_BODY_NAME}.1.{DOWN_BODY_NAME}.{i}") for i in range(layer)]])


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        out = self.style_loss(x, y)

        return out