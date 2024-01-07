import hydra
from torchvision import transforms
from model.velocity_transformer_ import VelocityFormer
from torch.utils.data import DataLoader
from dataset.auguments.data_auguments import CustomDataAugumentation
from dataset.trajectory_and_velocity import Trajectory_and_Velocity
import pytorch_lightning as pl

@hydra.main(config_path=f"../config", config_name="config")
def main(cfg):
    model = VelocityFormer(cfg.model, lr=cfg.lr)
    
    # 学習時にモデルの重みを保存する条件を指定
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='model/',
    )
    
    augumentation = CustomDataAugumentation()

    train_dataset = Trajectory_and_Velocity(cfg.dataset, mode="train", transform=augumentation)
    test_dataset = Trajectory_and_Velocity(cfg.dataset, mode="test")

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 学習の方法を指定
    trainer = pl.Trainer(
        default_root_dir="model/",
        gpus=1, 
        max_epochs=cfg.epocs,
        callbacks = [checkpoint]
    )
    trainer.fit(model, train_data_loader, test_data_loader)
    test = trainer.test(dataloaders=test_data_loader)
if __name__ == "__main__":
    main()
