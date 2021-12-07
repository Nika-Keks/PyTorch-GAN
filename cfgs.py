from dataclasses import dataclass

@dataclass
class DatasetInit:
    gt_path: str
    upscale: int
    n_rotation: int
    ext: str
    size: tuple[int, int] = (128, 128)
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

train_morowind_cfg = TrainData(
    datasetinit=DatasetInit(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\textureset1", 4, 4, ".bmp", (128, 128)),
    batch_size=8, 
    lr=(10**-4, 10**-6), 
    epochs=150, 
    weights_out_path=r"ganresnet\results_gan", 
    pretrained=True, 
    upscale=2,
    weights_path=(r"ganresnet\results_gan\g_epoch_109.pth", r"ganresnet\results_gan\d_epoch_109.pth"),
    start_epoch=109)

train_vaa_cfg = TrainData(
    datasetinit=DatasetInit(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\textureset2", 2, 4, ".png", (64, 64), "YCbCr"),
    batch_size=24, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r"ganresnet\results_epoch7_adv", 
    pretrained=True, 
    upscale=2,
    weights_path=(r"ganresnet\results_epoch7_adv\g_epoch_14.pth", r"ganresnet\results_epoch7_adv\d_epoch_14.pth"),
    start_epoch=14)

train_bsd_cfg = TrainData(
    datasetinit=DatasetInit(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\BSD100\Test_data\BSD100\HR", 2, 4, ".png", (64, 64)),
    batch_size=24, 
    lr=(10**-4, 10**-6), 
    epochs=30, 
    weights_out_path=r"ganresnet\results_bsd_gan", 
    pretrained=False, 
    upscale=2,
    weights_path=(None, None),
    start_epoch=0)

train_halo_cfg = TrainData(
    datasetinit=DatasetInit(r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\s64_fhalo\hr_64", 2, 4, ".png", (64, 64), "YCbCr"),
    batch_size=12, 
    lr=(10**-4, 10**-5), 
    epochs=100, 
    weights_out_path=r".\results\results_vgg_p_ssim", 
    pretrained=True, 
    upscale=2,
    weights_path=(r"results\results_vgg_p_ssim\g_epoch_15.pth", r"results\results_vgg_p_ssim\d_epoch_15.pth"),
    start_epoch=15)


############################## TEST CONGIGS #######################################

test_cfg1 = TestData(
    wights_path=r"results\results_ssim\g_epoch_15.pth",
    ext=".png",
    test_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\bic_x2",
    out_data_path=r"out_img",
    hr_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\gt",
    bic_sr_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\bicubic",
    calc_metrics=True
)

test_cfg2 = TestData(
    wights_path=r"results\results_ssim\g_epoch_15.pth",
    ext=".png",
    test_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\gt",
    out_data_path=r"out_img2",
    hr_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\vaa\gt",
    calc_metrics=False
)

big_test_cfg = TestData(
    wights_path=r"ganresnet\results_mgan\g_epoch_8.pth",
    ext=".bmp",
    test_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\big",
    out_data_path=r"ganresnet\out_img",
    hr_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\textureset1",
    calc_metrics=False
)


test_bsd_cfg = TestData(
    wights_path=r"ganresnet\results_bsd_gan\g_epoch_16.pth",
    ext=".png",
    test_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\Set5\LR\bicubic\x2",
    out_data_path=r"ganresnet\out_img",
    hr_data_path=r"C:\Users\Acer\Desktop\Documents\diploma\diploma\data\tests\Set5\HR",
    calc_metrics=True
)
