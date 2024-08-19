import os
import h5py
import numpy as np
import torch
import cv2
import argparse
from utils import load_data, find_near_index
from tqdm import tqdm
from event_utils import events_to_voxel_timesync_torch

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/M3ED/Falcon/Indoor/flight_1",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="falcon_indoor_flight_1",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--camera",
                    default="left",
                    help="which camera to use",
                    type=str)
    ap.add_argument("--method",
                    default="timesurface",
                    help="Event representation method",
                    type=str)
    ap.add_argument("--time_window",
                    default="200000",
                    help="length of time window",
                    type=int)
    ap.add_argument("--save_dir",
                    default="/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED",
                    help="Path to save preprocessed data",
                    type=str)
    args = ap.parse_args()


    root = args.dataset
    h5_file = f"{args.sequence}_data.h5"
    pose_file = f"{args.sequence}_pose_gt.h5"

    data_path = os.path.join(root, h5_file)
    pose_path = os.path.join(root, pose_file)

    data = h5py.File(data_path,'r')
    event_data_ref = load_data(data, sensor='prophesee', camera='left')
    event_data = load_data(data, sensor='prophesee', camera=args.camera)
    poses = h5py.File(pose_path,'r')
    ts_map_prophesee_left = poses['ts_map_prophesee_left']

    out_file = os.path.join(args.save_dir, args.sequence, f"event_frames_{args.method}_{args.time_window}", args.camera)
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    

    rows, cols = event_data['resolution'][1], event_data['resolution'][0]

    t_start = event_data['t'][0]

    for i in tqdm(range(len(ts_map_prophesee_left)-2)):
        idx_cur = int(ts_map_prophesee_left[i+1])
        t_ref = event_data_ref['t'][idx_cur]    # current timestamp

        idx_start, idx_cur, idx_end = find_near_index(event_data_ref['t'][idx_cur], event_data['t'], time_window=args.time_window)

        event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

        if os.path.exists(f"{out_file}/event_frame_{i:05d}.npy"):
            continue
        
        if args.method == "timesurface":
            # for id in range(idx_end-idx_start):
            for id in range(idx_cur - idx_start):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
        elif args.method == "bins":
            event_time_image = events_to_voxel_timesync_torch(torch.tensor(np.array(event_data['x'][idx_start:idx_cur], dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['y'][idx_start:idx_cur], dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['t'][idx_start:idx_cur], dtype=np.float32)), 
                                                                torch.tensor(np.array(event_data['p'][idx_start:idx_cur], dtype=np.float32)), 
                                                                B=10,
                                                                t0=event_data['t'][idx_start],
                                                                t1=event_data['t'][idx_cur],
                                                                np_ts=np.array(event_data['t'][idx_start:idx_cur], dtype=np.float32),
                                                                sensor_size=[rows, cols], temporal_bilinear=True)
        elif args.method == "SITS":
            r=6
            for id in range(idx_cur-idx_start):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 0]
                    patch = np.where(patch>=event_time_image[y,x,0], patch-1, patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 0] = patch
                    event_time_image[y, x, 0] = (2 * r + 1)**2
                else:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 1]
                    patch = np.where(patch>=event_time_image[y,x,1], patch-1, patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 1] = patch
                    event_time_image[y, x, 1] = (2 * r + 1)**2
            event_time_image[event_time_image < 0] = 0
        elif args.method == "ours":
            r = 6
            for id in range(idx_cur-idx_start):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 0]
                    patch = np.where(patch>=event_time_image[y,x,0], patch-(event_data['t'][idx]-patch)/15., patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 0] = patch
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    patch = event_time_image[y-r:y+r+1, x-r:x+r+1, 1]
                    patch = np.where(patch>=event_time_image[y,x,1], patch-(event_data['t'][idx]-patch)/15., patch)
                    event_time_image[y-r:y+r+1, x-r:x+r+1, 1] = patch
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        elif args.method == "ours_denoise":
            r = 6
            B = 15
            R = 1
            threshold = 0.3
            total_range = np.arange(idx_start, idx_cur)
            subsequences = np.array_split(total_range, B)
            for subseq in subsequences:
                mask = np.zeros((rows, cols), dtype=bool)
                for idx in subseq:
                    y, x = event_data['y'][idx], event_data['x'][idx]
                    if event_data['p'][idx] > 0:
                        patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                        patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
                        patch[patch<0] = 0
                        event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                        event_time_image[y, x, 0] = event_data['t'][idx]
                    else:
                        patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                        patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
                        patch[patch<0] = 0
                        event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                        event_time_image[y, x, 1] = event_data['t'][idx]
                    patch = event_time_image[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
                    valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
                    if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
                        mask[y, x] = True
                    else:
                        mask[y, x] = False
                event_time_image[mask] *= 0
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        else:
            raise "Method doesn't exit."


        now = np.array(event_time_image)
        np.save(f"{out_file}/event_frame_{i:05d}", now)


