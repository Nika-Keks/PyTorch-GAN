from dataclasses import dataclass

@dataclass
class DatasetInit:
    gt_path: str
    upscale: int
    n_rotation: int
    ext: str
    size: tuple = (128, 128)
    img_mode: str = "YCbCr"


    def tokwargs(self):
        return {
                "gt_path": self.gt_path,
                "upscale": self.upscale,
                "n_rotation": self.n_rotation,
                "ext": self.ext,
                "size": self.size
                }

@dataclass
class TrainData:
    """
    Class contain all configuration data for training GAN SRResNet model
    """
    datasetinit: DatasetInit
    batch_size: int
    lr: float
    epochs: int
    weights_out_path: str
    pretrained: bool
    start_epoch: int
    upscale: int = 2
    weights_path: str or tuple = None
    encoder_state: str = r"./autoencoder/results/w_mYCbCr/epoch_9.pth"

@dataclass
class TestData:
    """
    Class contain all configurating datat for testing GAN
    """
    wights_path: str
    ext: str
    test_data_path: str
    out_data_path: str
    hr_data_path: str
    calc_metrics: bool
    bic_sr_path: str = r""
    img_mode: str = "YCbCr"


################################### TRAIN CONGIGS #########################################


train_halo_cfg = TrainData(
    datasetinit=DatasetInit(r"./../data/fhalo", 2, 4, ".png", (64, 64), "YCbCr"),
    batch_size=8, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r"./results/w_enc_mse_adv", 
    pretrained=True, 
    upscale=2,
    weights_path=(r"./results/pretrained/g_epoch_7.pth", r"./results/pretrained/d_epoch_7.pth"),
    start_epoch=7)


############################## TEST CONGIGS #######################################

test_cfg1 = TestData(
    wights_path=r"./results/pretrained/g_epoch_7.pth",
    ext=".png",
    test_data_path=r"./../data/tests/...",
    out_data_path=r"out_img",
    hr_data_path=r"./../data/tmp/cfg1",
    calc_metrics=True,
    bic_sr_path=r""
)

test_cfg2 = TestData(
    wights_path=r"./results/pretrained/g_epoch_7.pth",
    ext=".png",
    test_data_path=r"./../data/tests/...",
    out_data_path=r"./../data/tmp/cfg2",
    calc_metrics=False,
    hr_data_path=r""
)
