import os
import torch
import glob
import hydra
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


class Trajectory_and_Velocity(Dataset):
    def __init__(self, cfg, mode="train", transform=None):
        self.root_dir = cfg.root_dir
        self.traj_dir = cfg.traj_dir
        self.control_dir = cfg.control_dir
        self.data_dir = []

        if mode=="train":
            for d in cfg.train_dir:
                self.data_dir.append(self.root_dir + "/" + d)
        else:
            for d in cfg.test_dir:
                self.data_dir.append(self.root_dir + "/" + d)

        self.dataset_for_loader = self._build(cfg)
        logger.info("Dataset dir num: {}".format(len(self.data_dir)))
        logger.info("Data num: {}".format(self.__len__()))
        
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.dataset_for_loader[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset_for_loader)
    
    def _build(self, cfg):
        dataset_for_loader = []
        for d in self.data_dir:
            files = glob.glob(d + "/" + self.traj_dir +"/*")
            for f in files:
                sample = {}
                basename = os.path.basename(f)
                # trajectory データを保存する
                traj = np.load(f)
                
                # 同時刻に保存された control データをロードする
                control = np.load(d + "/" + self.control_dir + "/" + basename)

                # trajectory を１次元ベクトルに変換する
                traj = self._parse_traj(traj,
                                        mode = cfg.trajectory.mode,
                                        interval = cfg.trajectory.interval,
                                        minimum_num = cfg.trajectory.minimum_num,
                                        point_num = cfg.trajectory.point_num,
                                        dtype = cfg.trajectory.dtype,
                                        )

                # trajectory が無効なデータの場合無視する
                if traj is None:
                    continue
                
                # 制御コマンドを1次元ベクトルに変換
                velocity = self._parse_velocity(control)

                sample["data"] = torch.from_numpy(traj)
                sample["label"] = torch.from_numpy(velocity)
                
                dataset_for_loader.append(sample)
        return dataset_for_loader

    # 制御コマンドデータを速度のベクトルに変換する
    def _parse_velocity(self, control, mode="regression"):
        # 回帰問題として速度の値を出力させる場合
        if mode == "regression":
            return np.array([int(control[0])], np.float32)

    # Trajectory points を1次元のベクトルに変換する
    def _parse_traj(self, traj: np.array,
                    mode: str = "degree",
                    interval: int = 10,
                    minimum_num: int = 150,
                    point_num: int= 10,
                    dtype="float16"
                    ):

        # 出力するフォーマットを指定
        if dtype =="float32":
             dtype=np.float32
        elif dtype =="int16":
             dtype=np.int16
        elif dtype =="int32":
             dtype=np.int32
        else:
            dtype=np.float16
        
        if len(traj) < minimum_num:
            return None
        
        points = []
        degrees = []
        for k in range(point_num):
            # 2つのtrajectory points から角度を出力する
            index = interval * k
            vec = traj[index][0:2] - traj[index - interval][0:2]
            base_vec = np.array([1, 0])
            degree = angle_between_vectors(vec, base_vec)
            degrees.append(degree)
            points.append(traj[index - interval][0:2]) # For debug
        #print(degrees) # For debug
        #plot_point(points) # For debug
        
        return np.array(degrees, dtype=dtype)

# 2つのベクトルの角度を算出する
def angle_between_vectors(v1, v2):
    # ベクトルの大きさを計算
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # ベクトルがゼロベクトルであるかどうかを確認
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        print("Warning: It's zero vector.")
        return 0

    # ドット積を計算
    dot_product = np.dot(v1, v2)

    # arccosを使用して角度を計算（ラジアンから度に変換）
    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_deg = np.degrees(angle_rad)
    
    if v1[1] < 0:
        angle_deg = 360 - angle_deg

    return angle_deg

def plot_point(points):
    import matplotlib.pyplot as plt
    points = np.array(points)
    plt.plot(points[:,0], points[:,1],marker='o')
    plt.plot(points[0,0], points[0,1],marker='o',color="r")
    #plt.ylim(0, 80000)
    #plt.xlim(0, 80000)
    plt.show()

@hydra.main(config_path=f"../../config", config_name="config")
def main(cfg):
    from auguments.data_auguments import CustomDataAugumentation
    cfg_dataset = cfg.dataset
    augumentation = CustomDataAugumentation()
    dataset = Trajectory_and_Velocity(cfg_dataset, transform=augumentation)
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    data_iter = iter(data_loader)
    for i in range(100):
        batch = next(data_iter)
        print(batch)
if __name__ == "__main__":
    main()