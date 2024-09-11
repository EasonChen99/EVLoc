import numpy as np
import os
import h5py
import argparse
import tqdm
import open3d as o3
from utils import load_map
import torch
import bisect

np.set_printoptions(threshold=np.inf)

def find_near_index(t_cur, ts):
    # Using bisect to find the closest index to t_cur
    pos_cur_t = bisect.bisect_left(ts, t_cur)
    if pos_cur_t == 0:
        idx_cur = pos_cur_t
    elif pos_cur_t == len(ts):
        idx_cur = pos_cur_t - 1
    else:
        idx_cur = pos_cur_t
    
    return idx_cur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--save_dir', type=str, default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/MVSEC', metavar='PARAMS', help='Main Directory to save all encoding results')
    parser.add_argument('--save_env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--data_path', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/MVSEC', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
    args = parser.parse_args()


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.data_path, args.save_env+"_data.hdf5")
    d_set = h5py.File(data_path, 'r')
    gray_images_ts = d_set['davis']['left']['image_raw_ts']
    gray_image = d_set['davis']['left']['image_raw']
    d_set = None

    gt_path = os.path.join(args.data_path, args.save_env+"_gt.hdf5")
    d_set = h5py.File(gt_path, 'r')
    poses = d_set['davis']['left']['odometry']
    poses_ts = d_set['davis']['left']['odometry_ts']
    d_set = None

    T_cam0_lidar = np.array(
                    [[ 0.01140786, -0.99985524,  0.01262402,  0.03651404],
                     [ 0.04312291, -0.01212116, -0.99899624, -0.08637514],
                     [ 0.99900464,  0.0119408 ,  0.04297839, -0.12882002],
                     [ 0.        ,  0.        ,  0.        ,  1.        ]]  
                    )

    print(gray_images_ts.shape, poses_ts.shape)
    print(gray_images_ts[:10]/1e5)
    print(poses_ts[:10]/1e5)
    print(gray_images_ts[1] - gray_images_ts[0], gray_images_ts[2]-gray_images_ts[1], gray_images_ts[3]-gray_images_ts[2])
    print(poses_ts[1]-poses_ts[0],poses_ts[2]-poses_ts[1],poses_ts[3]-poses_ts[2])

    save_path = f"{args.save_dir}/{args.save_env}/event_frames_gray____/left"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(len(poses_ts)):
        gray_idx = find_near_index(poses_ts[idx], gray_images_ts)

        np.save(f"{save_path}/event_frame_{idx:05d}.npy", gray_image[gray_idx,:,:])
        # import cv2
        # cv2.imwrite(f"{save_path}/event_frame_{idx:05d}.png", gray_image[gray_idx,:,:])

