import os
import sys
import h5py
import numpy as np
import cv2
import open3d as o3
import torch
from tqdm import tqdm
from utils import load_data, load_map, pc_visualize, depth_generation, find_near_index
from skimage.draw import line as line_function

import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", 
                    default="/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/M3ED/original/Falcon/Indoor/flight_3",
                    help="Root path to the dataset", 
                    type=str)
    args.add_argument("--sequence",
                    default="falcon_indoor_flight_3",
                    help="Sequence name for processing",
                    type=str)
    args.add_argument("--method",
                    default="TS",
                    help="Event representation method",
                    type=str)
    args.add_argument("--idx",
                    default=200,
                    type=int)
    args = args.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    root = args.dataset
    name = args.sequence
    h5_file = f"{name}_data.h5"
    pose_file = f"{name}_depth_gt.h5"
    pc_file = f"{name}_global.pcd"

    data_path = os.path.join(root, h5_file)
    pose_path = os.path.join(root, pose_file)
    pc_path = os.path.join(root, pc_file)

    data = h5py.File(data_path,'r')
    rgb_data = load_data(data, sensor='ovc', camera='rgb') 
    rgbs = rgb_data['img'][:]
    rgb_ts = rgb_data['ts'][:]
    event_data = load_data(data, sensor='prophesee', camera='left')
    T_to_prophesee_left = torch.tensor(event_data['T_to_prophesee_left'])
    # # pose load
    poses = h5py.File(pose_path,'r')
    depths = poses['depth/prophesee/left']
    Cn_T_C0 = poses['Cn_T_C0']                                                  # 570 4 4
    Ln_T_L0 = poses['Ln_T_L0']                                                  # 570 4 4
    pose_ts = poses['ts']                                                       # 570
    ts_map_prophesee_left = poses['ts_map_prophesee_left']                      # 570 index

    if args.method == "all":
        idxs = range(len(ts_map_prophesee_left)-2)
    idxs = [args.idx]
    
    for i in idxs:
        idx_cur = int(ts_map_prophesee_left[i+1])
        t_ref = event_data['t'][idx_cur]
        idx_start, idx_cur, idx_end = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=200000)
        rows, cols = event_data['resolution'][1], event_data['resolution'][0]
        print(idx_start, idx_cur, idx_end)
        print(event_data['t'][idx_start], event_data['t'][idx_cur], event_data['t'][idx_end])
        if args.method == "TS":
            # # Time surface
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
        elif args.method == "SITS":
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            r=6
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch = np.where(patch>=event_time_image[y,x,0], patch-1, patch)
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    event_time_image[y, x, 0] = (2 * r + 1)**2
                else:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch = np.where(patch>=event_time_image[y,x,1], patch-1, patch)
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    event_time_image[y, x, 1] = (2 * r + 1)**2
            event_time_image[event_time_image < 0] = 0
        elif args.method == "CM":
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            r = 6
            threshold = 0.32
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch_valid = patch[patch>0]
                    if len(patch_valid) <= 1:
                        contrast = 0.
                    else:
                        patch_valid = (patch_valid - np.min(patch_valid))/(np.max(patch_valid) - np.min(patch_valid))
                        contrast = np.sqrt(np.var(patch_valid))
                    if contrast>threshold:
                        continue
                    patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch_valid = patch[patch>0]
                    if len(patch_valid) <= 1:
                        contrast = 0.
                    else:
                        patch_valid = (patch_valid - np.min(patch_valid))/(np.max(patch_valid) - np.min(patch_valid))
                        contrast = np.sqrt(np.var(patch_valid))
                    if contrast>threshold:
                        continue
                    patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        elif args.method == "Ours":
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            r = 6
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
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
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        elif args.method == "Ours_Denoise":
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            r = 6
            B = 1 # 15
            R = 1
            threshold = 0.7 # 0.3
            total_range = np.arange(idx_start, idx_cur)
            subsequences = np.array_split(total_range, B)
            for subseq in subsequences:
                mask = np.zeros((rows, cols), dtype=bool)
                for idx in tqdm(subseq):
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
        elif args.method == "Ours_Line":
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            r = 6
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                    
                    valid_pixel_y, valid_pixel_x = np.where(patch>0)
                    line_valid_pixel_list = [list(zip(line_function(valid_pixel_y[i], valid_pixel_x[i], 6, 6))) for i in range(len(valid_pixel_x))]
                    # _ = [print(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
                    line_count = [len(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
                    line_valid_count = [(patch[line_index_yx[0][0], line_index_yx[1][0]]>0).sum() for line_index_yx in line_valid_pixel_list]
                    weight = np.asarray(line_count) / np.asarray(line_valid_count)
                    patch[valid_pixel_y, valid_pixel_x] -=  weight * (event_data['t'][idx]-patch[valid_pixel_y, valid_pixel_x])/20.
                    
                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    event_time_image[y, x, 0] = event_data['t'][idx]
                else:
                    patch = event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/10., patch)
                    
                    valid_pixel_y, valid_pixel_x = np.where(patch>0)
                    line_valid_pixel_list = [list(zip(line_function(valid_pixel_y[i], valid_pixel_x[i], 6, 6))) for i in range(len(valid_pixel_x))]
                    # _ = [print(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
                    line_count = [len(line_index_yx[0][0]) for line_index_yx in line_valid_pixel_list]
                    line_valid_count = [(patch[line_index_yx[0][0], line_index_yx[1][0]]>0).sum() for line_index_yx in line_valid_pixel_list]
                    weight = np.asarray(line_count) / np.asarray(line_valid_count)
                    patch[valid_pixel_y, valid_pixel_x] -=  weight * (event_data['t'][idx]-patch[valid_pixel_y, valid_pixel_x])/20.

                    patch[patch<0] = 0
                    event_time_image[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    event_time_image[y, x, 1] = event_data['t'][idx]
            event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
            event_time_image[event_time_image < 0] = 0
        elif args.method == "Bins":
            # # Voxel
            from event_utils import events_to_voxel_timesync_torch
            event_time_image = events_to_voxel_timesync_torch(torch.tensor(np.array(event_data['x'][idx_start:idx_end], dtype=np.float32)), 
                                                            torch.tensor(np.array(event_data['y'][idx_start:idx_end], dtype=np.float32)), 
                                                            torch.tensor(np.array(event_data['t'][idx_start:idx_end], dtype=np.float32)), 
                                                            torch.tensor(np.array(event_data['p'][idx_start:idx_end], dtype=np.float32)), 
                                                            B=10,
                                                            t0=event_data['t'][idx_start],
                                                            t1=event_data['t'][idx_end],
                                                            np_ts=np.array(event_data['t'][idx_start:idx_end], dtype=np.float32),
                                                            sensor_size=[rows, cols], temporal_bilinear=True) # 10 720 1280
        elif args.method == "all":
            time_surface = np.zeros((rows, cols, 2), dtype=np.float32)
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    time_surface[y, x, 0] = event_data['t'][idx]
                else:
                    time_surface[y, x, 1] = event_data['t'][idx]
            time_surface[time_surface > 0] -= event_data['t'][idx_start]
            time_surface = np.array(time_surface)
            time_surface_disp = np.concatenate((np.zeros([time_surface.shape[0], time_surface.shape[1], 1]), time_surface), axis=2)
            time_surface_disp = (time_surface_disp / np.max(time_surface_disp) * 255).astype(np.uint8)
            cv2.imwrite(f'./visualization/event_frame_{i:05d}_0_TS.png', time_surface_disp)

            SI_time_surface = np.zeros((rows, cols, 2), dtype=np.float32)
            r=6
            for id in tqdm(range(idx_cur-idx_start)):
                idx = int(id + idx_start)
                y, x = event_data['y'][idx], event_data['x'][idx]
                if event_data['p'][idx] > 0:
                    patch = SI_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
                    patch = np.where(patch>=SI_time_surface[y,x,0], patch-1, patch)
                    SI_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
                    SI_time_surface[y, x, 0] = (2 * r + 1)**2
                else:
                    patch = SI_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
                    patch = np.where(patch>=SI_time_surface[y,x,1], patch-1, patch)
                    SI_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
                    SI_time_surface[y, x, 1] = (2 * r + 1)**2
            SI_time_surface[SI_time_surface < 0] = 0
            SI_time_surface = np.array(SI_time_surface)
            SI_time_surface_disp = np.concatenate((np.zeros([SI_time_surface.shape[0], SI_time_surface.shape[1], 1]), SI_time_surface), axis=2)
            SI_time_surface_disp = (SI_time_surface_disp / np.max(SI_time_surface_disp) * 255).astype(np.uint8)
            cv2.imwrite(f'./visualization/event_frame_{i:05d}_1_SITS.png', SI_time_surface_disp)

            # S_time_surface = np.zeros((rows, cols, 2), dtype=np.float32)
            # r = 6
            # B = 15
            # R = 1
            # threshold = 0.3
            # total_range = np.arange(idx_start, idx_cur)
            # subsequences = np.array_split(total_range, B)
            # for subseq in subsequences:
            #     mask = np.zeros((rows, cols), dtype=bool)
            #     for idx in tqdm(subseq):
            #         y, x = event_data['y'][idx], event_data['x'][idx]
            #         if event_data['p'][idx] > 0:
            #             patch = S_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0]
            #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
            #             patch[patch<0] = 0
            #             S_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 0] = patch
            #             S_time_surface[y, x, 0] = event_data['t'][idx]
            #         else:
            #             patch = S_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
            #             patch = np.where(patch>0, patch-(event_data['t'][idx]-patch)/15., patch)
            #             patch[patch<0] = 0
            #             S_time_surface[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1] = patch
            #             S_time_surface[y, x, 1] = event_data['t'][idx]
            #         patch = S_time_surface[max(0, y-R):y+R+1, max(0, x-R):x+R+1, :]
            #         valid_count = ((patch[:, :, 0] > 0) | (patch[:, :, 1] > 0)).sum()
            #         if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
            #             mask[y, x] = True
            #         else:
            #             mask[y, x] = False
            #     S_time_surface[mask] *= 0
            # S_time_surface[S_time_surface > 0] -= event_data['t'][idx_start]
            # S_time_surface[S_time_surface < 0] = 0
            # S_time_surface = np.array(S_time_surface)
            # S_time_surface_disp = np.concatenate((np.zeros([S_time_surface.shape[0], S_time_surface.shape[1], 1]), S_time_surface), axis=2)
            # S_time_surface_disp = (S_time_surface_disp / np.max(S_time_surface_disp) * 255).astype(np.uint8)
            # cv2.imwrite(f'./visualization/event_frame_{idx:05d}_2_Ours.png', S_time_surface_disp)
        else:
            event_time_image = np.load(f"/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED/falcon_indoor_flight_1/event_frames_ours_pre_100000/left/event_frame_{args.idx:05d}.npy")
        
        if args.method != "all":
            event_time_image = np.array(event_time_image)
            event_time_image_disp = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
            event_time_image_disp = (event_time_image_disp / np.max(event_time_image_disp) * 255).astype(np.uint8)
            event_time_image_disp = event_time_image_disp[60:660, 160:1060, :]
            cv2.imwrite(f'./event_frame_{args.idx:05d}_{args.method}.png', event_time_image_disp)
    
    
    
    # event_time_image_denoise = event_time_image_disp
    # event_time_image_denoise = cv2.medianBlur(event_time_image_disp, 3)
    # # event_time_image_denoise = cv2.bilateralFilter(event_time_image_disp, 9, 150, 150)
    # # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (7,7), 0)
    # # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (5,5), 0)
    # # R = 3
    # # threshold = 0.3
    # # valid = (event_time_image_denoise[:, :, 1] > 0) | (event_time_image_denoise[:, :, 2] > 0)
    # # valid_pixel_y, valid_pixel_x = np.where(valid)
    # # mask = np.zeros((rows, cols), dtype=bool)
    # # for y, x in zip(valid_pixel_y, valid_pixel_x):
    # #     patch = event_time_image_denoise[max(0, y-r):y+r+1, max(0, x-r):x+r+1, :]
    # #     valid_count = ((patch[:, :, 1] > 0) | (patch[:, :, 2] > 0)).sum()
    # #     if valid_count / (patch.shape[0]*patch.shape[1]) < threshold:
    # #         mask[y, x] = True
    # # event_time_image_denoise[mask] *= 0
    # # event_time_image_denoise = cv2.medianBlur(event_time_image_denoise, 3)
    # event_time_image_denoise = (event_time_image_denoise / np.max(event_time_image_denoise) * 255).astype(np.uint8)
    # cv2.imwrite(f'./event_frame_{args.method}_denoise.png', event_time_image_denoise)



    # # idx_cur = int(ts_map_prophesee_left[i+1])
    # # t_ref = event_data['t'][idx_cur]
    # # idxes = []
    # # idx_start_1, idx_start_6, idx_end_10 = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=200000)
    # # idx_start_2, _, idx_start_10 = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=160000)
    # # idx_start_3, _, idx_start_9 = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=120000)
    # # idx_start_4, _, idx_start_8 = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=80000)
    # # idx_start_5, _, idx_start_7 = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=40000)
    # # idxes.append(idx_start_1)
    # # idxes.append(idx_start_2)
    # # idxes.append(idx_start_3)
    # # idxes.append(idx_start_4)
    # # idxes.append(idx_start_5)
    # # idxes.append(idx_start_6)
    # # idxes.append(idx_start_7)
    # # idxes.append(idx_start_8)
    # # idxes.append(idx_start_9)
    # # idxes.append(idx_start_10)
    # # idxes.append(idx_end_10)
    # # rows, cols = event_data['resolution'][1], event_data['resolution'][0]
    # # # # Time surface
    # # for sub in range(10):
    # #     event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
    # #     print(sub, idxes[sub], idxes[sub+1])
    # #     for id in tqdm(range(idxes[sub+1]-idxes[sub])):
    # #         idx = int(id + idxes[sub])
    # #         y, x = event_data['y'][idx], event_data['x'][idx]
    # #         if event_data['p'][idx] > 0:
    # #             event_time_image[y, x, 0] = event_data['t'][idx]
    # #         else:
    # #             event_time_image[y, x, 1] = event_data['t'][idx]
    # #     event_time_image[event_time_image > 0] -= event_data['t'][idxes[sub]]
    # #     event_time_image = np.array(event_time_image)
    # #     event_time_image_disp = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
    # #     cv2.imwrite(f'./event_frame_{sub:02d}.png', (event_time_image_disp / np.max(event_time_image_disp) * 255).astype(np.uint8))

    # # # # pc map load
    # # vox_map = load_map(pc_path, device)
    # # print(f'load pointclouds finished! {vox_map.shape[1]} points')
    # # pose = torch.tensor(Ln_T_L0[i+1], device=device, dtype=torch.float32)
    # # pose_inv = pose.inverse()
    # # local_map = vox_map.clone()
    # # local_map = torch.matmul(pose, local_map)
    # # indexes = local_map[0, :] > -1.
    # # indexes = indexes & (local_map[0, :] < 10.)
    # # indexes = indexes & (local_map[1, :] > -5.)
    # # indexes = indexes & (local_map[1, :] < 5.)
    # # local_map = local_map[:, indexes]
    # # prophesee_left_T_lidar = torch.tensor(data["/ouster/calib/T_to_prophesee_left"], device=device, dtype=torch.float32)
    # # local_map = torch.matmul(prophesee_left_T_lidar, local_map)
    # # # pc_visualize(local_map.t().cpu().numpy())
    # # image_size = tuple(event_data['resolution'][[1, 0]])
    # # calib = torch.tensor(event_data['intrinsics'], device=device, dtype=torch.float)
    # # sparse_depth = depth_generation(local_map, image_size,
    # #                                 calib, 3., 5, device=device)
    # # sparse_depth_disp = (sparse_depth / torch.max(sparse_depth) * 255).cpu().numpy().astype(np.uint8)
    # # cv2.imwrite('./LiDAR_projection.png', sparse_depth_disp[0, :])


    # # # rgb load
    # # ts_pose = pose_ts[i+1]
    # # index = (ts_pose - rgb_ts[0]) // 40000
    # # rgb = rgbs[int(index), ...]
    # # cv2.imwrite('./rgb.png', rgb)

    # # prophesee_left_T_rgb = torch.tensor(rgb_data['T_to_prophesee_left'], device=device, dtype=torch.float32)
    # # local_map = torch.matmul(prophesee_left_T_rgb.inverse(), local_map)
    # # image_size = tuple(rgb_data['resolution'][[1, 0]])
    # # calib = torch.tensor(rgb_data['intrinsics'], device=device, dtype=torch.float)
    # # sparse_depth = depth_generation(local_map, image_size,
    # #                                 calib, 3., 5, device=device)
    # # sparse_depth_disp = (sparse_depth / torch.max(sparse_depth) * 255).cpu().numpy().astype(np.uint8)
    # # cv2.imwrite('./LiDAR_projection_rgb.png', sparse_depth_disp[0, :])
