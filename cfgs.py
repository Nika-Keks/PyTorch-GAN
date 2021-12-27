from dataclasses import dataclass
from PIL import Image

@dataclass
class DatasetInit:
    gt_path: str
    upscale: int
    n_rotation: int
    ext: str
    size: tuple = (128, 128)
    img_mode: str = "YCbCr"
    resample_mode = Image.BICUBIC


    def tokwargs(self):
        return {
                "gt_path": self.gt_path,
                "upscale": self.upscale,
                "n_rotation": self.n_rotation,
                "ext": self.ext,
                "size": self.size,
                "img_mode": self.img_mode,
                "resampling_mode": self.resample_mode
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
    encoder_state: str
    upscale: int = 2
    weights_path: str or tuple = None

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
    batch_size=4, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r"./results/w_tests", 
    pretrained=True, 
    upscale=2,
    encoder_state=r"./autoencoder/results/w_res_maenet1/epoch_5_it_60000.pth",
    weights_path=(r"./results/pretrained/g_epoch_7.pth", r"./results/pretrained/d_epoch_7.pth"),
    start_epoch=7)


train_test_cfg = TrainData(
    datasetinit=DatasetInit(r"./../data/tests/fhalo_test", 2, 4, ".png", (64, 64), "YCbCr"),
    batch_size=4, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r"./results/w_tests", 
    pretrained=True, 
    upscale=2,
    encoder_state=r"./autoencoder/results/w_maenet_pret_2/epoch_4_it_180000.pth",
    weights_path=(r"./results/pretrained/g_epoch_7.pth", r"./results/pretrained/d_epoch_7.pth"),
    start_epoch=7)


############################## TEST CONGIGS #######################################

test_cfg1 = TestData(
    wights_path=r"./results/pretrained/g_epoch_7.pth",
    ext=".png",
    test_data_path=r"./../data/tests/fhalo_test",
    out_data_path=r"./../data/tmp/cfg1",
    bic_sr_path=r"",
    calc_metrics=False,
    hr_data_path=r""
)

test_cfg2 = TestData(
    wights_path=r"./results/w_tests/g_epoch_99.pth",
    ext=".png",
    test_data_path=r"./../data/tests/fhalo_test",
    out_data_path=r"./../data/tmp/cfg2",
    calc_metrics=False,
    hr_data_path=r""
)
