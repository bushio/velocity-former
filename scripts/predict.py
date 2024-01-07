import hydra
import os
from torchvision import transforms
from model.velocity_transformer_ import VelocityFormer
from torch.utils.data import DataLoader
from dataset.auguments.data_auguments import CustomDataAugumentation
from dataset.trajectory_and_velocity import Trajectory_and_Velocity
import pytorch_lightning as pl

@hydra.main(config_path=f"../config", config_name="config")
def main(cfg):
    work_dir = hydra.utils.get_original_cwd()
    os.chdir(work_dir)
    ckpt = cfg.ckpt
    if ckpt is None:
        print("Please set traied mode as ckpt.")
        exit()
    model = VelocityFormer(cfg.model, lr=cfg.lr)
    model = model.load_from_checkpoint(ckpt)
    test_dataset = Trajectory_and_Velocity(cfg.dataset, mode="test")
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for x in test_data_loader:
        output = model(x["data"])
        print(output[0], x["label"])

if __name__ == "__main__":
    main()
