import torch

from aenet import AEnet
from torch import nn


class EncoderLoss(nn.Module):

    def __init__(self, model_state_dict_path: str, criterias_mode: str = "l2") -> None:
        """Loss based on encoder represantation

        Args:
            model_state_dict_path (str): path to train wall model state dict
            criterias_mode (str, optional): criterias for representation, ["l1", "l2"]. Defaults to "l2".

        Raises:
            NotImplementedError: if criterias mode not in ["l1", "l2"]
        """
        super().__init__()

        autoenc = AEnet()
        autoenc.load_state_dict(torch.load(model_state_dict_path)["model"])
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