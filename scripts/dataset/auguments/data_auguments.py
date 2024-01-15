import numpy as np
from torchvision import transforms

class CustomDataAugumentation:
    def __init__(self, label_type="velocity"):
        self.transforms = [RotatePoint(label_type=label_type)]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    

class RotatePoint():
    def __init__(self, label_type="velocity", mode_num=3):
        self.mode_num = mode_num
        self.label_type = label_type
    def __call__(self, sample):
        mode = np.random.randint(self.mode_num)
        if mode == 0:
            # オリジナルの値を返す
            sample["data"] = sample["data"]
        elif mode == 1:
            # y軸対称に変換
            tensor = sample["data"] 
            tensor = 180 - tensor
            tensor[tensor < 0] +=360
            sample["data"] = tensor
            if self.label_type == "steering":
                sample["label"] = sample["label"] * -1
        else:
            # x軸対称に変換
            sample["data"] = 360 - sample["data"]
            if self.label_type == "steering":
                sample["label"] = sample["label"] * -1
        return sample
