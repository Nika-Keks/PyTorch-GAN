import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt





def select_err(file_path: str, part: float = None) -> np.array:
    file = open(file_path, "r")
    
    if part is None:
        return np.array([float(x) for x in file]), None

    file_len = sum([1 for _ in open(file_path, "r")])
    n_points = file_len
    n_points = int(n_points * part)
    if n_points == 0:
        print(f"file {file_path} empty")
        return None, None

    n_mean = file_len // n_points

    points = np.array([np.mean([float(line) for _, line in zip(range(n_mean), file)]) for _ in range(n_points)])
    
    return points, n_points
            


if __name__ == "__main__":

    #losses_path = r"./autoencoder/loss_data/l_res_maenet1"
    losses_path = r"./srgan/loss_data"

    for lfile_name in os.listdir(losses_path):

        if lfile_name.endswith(".loss"):
            name = lfile_name[0:lfile_name.find(".loss")]
        else:
            continue
        
        x, n_epochs = select_err(os.path.join(losses_path, lfile_name), 0.01)
        if x is None:
            continue
        plt.title(name)
        plt.plot(x)
        plt.xlabel("countdown")
        plt.ylabel("mean loss function value for 100 iterations")
        plt.plot(pd.Series(x).rolling(window=501, min_periods=1, center=True).median())
        plt.grid()
        os.makedirs("plots", exist_ok=True)
        plt.show()
        plt.savefig(os.path.join("plots", f"{name}.png"))
        plt.close()
        