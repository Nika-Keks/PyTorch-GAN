import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt





def select_err(file_path: str, n_samples: int, part: float) -> np.array:
    file = open(file_path, "r")
    file_len = sum([1 for _ in open(file_path, "r")])
    n_epochs = file_len // n_samples
    n_points = int(n_epochs * n_samples * part)
    n_mean = file_len // n_points

    points = np.array([np.mean([float(line) for _, line in zip(range(n_mean), file)]) for _ in range(n_points)])
    
    return points, n_epochs
            


if __name__ == "__main__":

    n_samples = 1
    losses_path = os.path.join("loss_data", "l_ssim")

    for lfile_name in os.listdir(losses_path):

        if lfile_name.endswith(".loss"):
            name = lfile_name[0:lfile_name.find(".loss")]
        else:
            continue
        
        x, n_epochs = select_err(os.path.join(losses_path, lfile_name), n_samples, 0.0001)
        plt.title(name)
        plt.plot(x)
        plt.plot(pd.Series(x).rolling(window=501, min_periods=1, center=True).median())
        plt.grid()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", f"{name}.png"))
        plt.close()
        