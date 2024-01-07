import numpy as np
from torchvision import transforms

class CustomDataAugumentation:
    def __init__(self):
        self.transforms = [RotatePoint()]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    

class RotatePoint():
    def __call__(self, sample):
        mode = np.random.randint(3)
        if mode == 0:
            # オリジナルの値を返す
            sample["data"] = sample["data"]
        elif mode == 1:
            # y軸対称に変換
            tensor = sample["data"] 
            tensor = 180 - tensor
            tensor[tensor < 0] +=360
            sample["data"] = tensor
        else:
            # x軸対称に変換
            sample["data"] = 360 - sample["data"]
        return sample
