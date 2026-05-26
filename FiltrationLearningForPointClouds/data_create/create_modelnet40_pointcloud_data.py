import os
import torch
import numpy as np
import random
from tqdm import tqdm

from lib.modelnet_process import read_off, PointSampler

sampling_time = 1
sampling_point_num = 2000
cls_num = 40
data_per_cls = 100
noise_level = 0.1

filename_data = f"data/ModelNetNoisy01_C={cls_num},N={data_per_cls},T={sampling_time},K={sampling_point_num}_data"
filename_label = f"data/ModelNetNoisy01_C={cls_num},N={data_per_cls},T={sampling_time},K={sampling_point_num}_label"

MODELNET40_ROOT = os.path.expanduser("~/PPM_Expected_FL_2/ModelNet40")

cls_list = sorted([d for d in os.listdir(MODELNET40_ROOT)
                   if os.path.isdir(os.path.join(MODELNET40_ROOT, d))])
assert len(cls_list) == cls_num, f"Expected {cls_num} classes, got {len(cls_list)}"
train_cls_list = cls_list[:cls_num]
print(f"Classes ({len(train_cls_list)}): {train_cls_list}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

pointcloud_list = []
for cls in tqdm(train_cls_list):
    cls_dir = os.path.join(MODELNET40_ROOT, cls, "train")
    off_files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".off")])
    n_meshes = len(off_files)
    if n_meshes >= data_per_cls:
        chosen_files = off_files[:data_per_cls]
    else:
        chosen_files = [off_files[i % n_meshes] for i in range(data_per_cls)]

    for fname in chosen_files:
        fpath = os.path.join(cls_dir, fname)
        with open(fpath, 'r') as f:
            verts, faces = read_off(f)
        for j in range(sampling_time):
            pointcloud = PointSampler(sampling_point_num)((verts, faces))
            avg = np.mean(pointcloud, axis=0)
            std = np.std(pointcloud, axis=0)
            pointcloud = (pointcloud - avg.reshape(1, 3)) / np.maximum(std, 1e-8).reshape(1, 3)
            pointcloud_list.append(torch.tensor(pointcloud).to(torch.float32))

data = torch.stack(pointcloud_list, axis=0)
data = data + torch.normal(0., noise_level, size=data.shape)
label = torch.tensor([i // (data.shape[0] // cls_num) for i in range(data.shape[0])]).to(torch.long)

data = data[:, random.sample(range(data.shape[1]), k=data.shape[1]), :]

print(f"data shape: {data.shape}, label shape: {label.shape}")
print(f"label distribution: {torch.bincount(label).tolist()}")

torch.save(data, filename_data)
torch.save(label, filename_label)
print(f"Saved -> {filename_data}")
print(f"Saved -> {filename_label}")
