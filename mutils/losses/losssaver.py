import os
import sys

__all__ = ["LossSaver"]

class LossSaver():

    def __init__(self, name: str, path: str = os.path.join("loss_data","losses")) -> None:
        file_path = os.path.join(path, f"{name}.loss")
        os.makedirs(path, exist_ok=True)
        self.file = open(file_path, "a")
        

    def __call__(self, x: float):
        self.file.write(f"{x:.32f}")
        self.file.write("\n")
