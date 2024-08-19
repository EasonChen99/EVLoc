import os
import h5py
import numpy as np
import cv2
import open3d as o3
import torch
from tqdm import tqdm
from utils import load_data, load_map, pc_visualize, depth_generation, find_near_index


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

name = "falcon_indoor_flight_2"
root = "/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/M3ED/Falcon/Indoor/flight_2"
h5_file = f"{name}_data.h5"
pose_file = f"{name}_depth_gt.h5"
camera = 'right'

data_path = os.path.join(root, h5_file)
pose_path = os.path.join(root, pose_file)

data = h5py.File(data_path,'r')
rgb_data = load_data(data, sensor='ovc', camera=camera) 
rgb_ts = rgb_data['ts'][:]
poses = h5py.File(pose_path,'r')
pose_ts = poses['ts'][:]
Ln_T_L0 = poses['Ln_T_L0']

save_path = f"/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED/{name}/event_frames_gray____/{camera}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

rgbs = rgb_data['img'][:]
for i in tqdm(range(pose_ts.shape[0]-1)):
    index = int((pose_ts[i] - rgb_ts[0]) // 40000)
    rgb = rgbs[index, ...]
    # cv2.imwrite(f"{save_path}/event_frame_{i:05d}.png", rgb) 
    np.save(f"{save_path}/event_frame_{i:05d}.npy", rgb)