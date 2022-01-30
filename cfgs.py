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
    resample_mode = Image.NEAREST


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
    name: str
    batch_size: int
    lr: float
    epochs: int
    weights_out_path: str
    pretrained: bool
    start_epoch: int
    encoder_state: str
    upscale: int = 2
    weights_path: str or tuple = None
    loss_data_path: str = r"./srgan/loss_data"

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
    datasetinit=DatasetInit(r"./../data/chalo", 4, 4, ".png", (64, 64), "YCbCr"),
    name="chalo_up_mse_pret7",
    batch_size=4, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r"./srgan/results", 
    pretrained=True, 
    upscale=2,
    encoder_state=r"./autoenc/results/w_l_3sim_0mse_s_64x64_m_ycbcr/epoch_6.pth",
    weights_path=(r"./srgan/results/w_chalo_up_mse_pret7/g_epoch_18_1.pth", r"./srgan/results/w_chalo_up_mse_pret7/d_epoch_18_1.pth"),
    start_epoch=17)

train_moro_cfg = TrainData(
    datasetinit=DatasetInit(r"./../data/morrowind", 4, 4, ".bmp", (64, 64), "YCbCr"),
    name="moro_up_mse_pret7",
    batch_size=4, 
    lr=(10**-4, 10**-5), 
    epochs=20, 
    weights_out_path=r"./srgan/results", 
    pretrained=True, 
    upscale=2,
    encoder_state=r"./autoenc/results/w_l_3sim_0mse_s_64x64_m_ycbcr/epoch_6.pth",
    weights_path=(r"./srgan/results/pretrained/g_epoch_7.pth", r"./srgan/results/pretrained/d_epoch_7.pth"),
    start_epoch=7)

train_test_cfg = TrainData(
    datasetinit=DatasetInit(r"./../data/chalo", 4, 4, ".png", (64, 64), "YCbCr"),
    name="test",
    batch_size=4, 
    lr=(10**-4, 10**-5), 
    epochs=200, 
    weights_out_path=r"./srgan/results", 
    pretrained=True, 
    upscale=2,
    encoder_state=r"./autoenc/results/w_l_3sim_0mse_s_64x64_m_ycbcr/epoch_6.pth",
    weights_path=(r"./srgan/results/pretrained/g_epoch_7.pth", r"./srgan/results/pretrained/d_epoch_7.pth"),
    start_epoch=7)


############################## TEST CONGIGS #######################################

test_cfg1 = TestData(
    wights_path=r"./srgan/results/w_halo_up_mse_pret7/g_epoch_10.pth",
    ext=".png",
    test_data_path=r"./../data/tests/fhalo_test",
    out_data_path=r"./../data/tmp/cfg1",
    bic_sr_path=r"",
    calc_metrics=False,
    hr_data_path=r""
)

test_cfg2 = TestData(
    wights_path=r"./srgan/results/w_test/g_epoch_11.pth",
    ext=".png",
    test_data_path=r"./../data/tests/fhalo_test",
    out_data_path=r"./../data/tmp/cfgt",
    calc_metrics=False,
    hr_data_path=r"",
    img_mode="YCbCr"
)
