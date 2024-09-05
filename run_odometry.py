import csv
import open3d as o3
import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import visibility as visibility
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse
import cv2
import sys
import h5py

from core.camera_model import CameraModel
from core.backbone import Backbone_Event_Offset_RT
from core.utils_point import quaternion_from_matrix, to_rotation_matrix, overlay_imgs
from core.utils import count_parameters
from core.flow2pose import Flow2Pose, warp
from core.quaternion_distances import quaternion_distance
from core.data_preprocess import Data_preprocess
from core.flow_viz import flow_to_image

def get_calib_m3ed(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([1034.86278431, 1033.47800271, 629.70125104, 357.60071019])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([1034.39696079, 1034.73607278, 636.27410756, 364.73952748])
    elif sequence in ['spot_indoor_building_loop', 'spot_indoor_obstacles', 'spot_indoor_stairs']:
        return torch.tensor([1032.0231, 1031.9229,  635.7985,  363.7983])
    elif sequence in ['spot_outdoor_day_srt_under_bridge_1']:
        return torch.tensor([1030.29161359, 1030.9024083, 634.79835424, 368.11576903])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([1033.22781771, 1032.05548869, 631.84536312, 360.7175681])
    elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        return torch.tensor([1034.61302587, 1034.83604567, 638.12992827, 366.88002829])
    elif sequence in ['car_urban_day_penno_small_loop']:
        return torch.tensor([1031.36879978, 1031.06491961, 634.87768084, 367.62546105])
    elif sequence in ['car_urban_night_penno_small_loop']:
        return torch.tensor([1030.46186128, 1029.51180204, 635.69022466, 364.32444857])
    else:
        raise TypeError("Sequence Not Available")

def get_velo2cam_m3ed(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 0.0076, -0.9996, -0.0259,  0.0482],
                             [-0.2454,  0.0233, -0.9692, -0.2190],
                             [ 0.9694,  0.0137, -0.2451, -0.2298],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[-1.0447e-04, -9.9970e-01, -2.4339e-02,  5.0678e-02],
                             [-2.4484e-01,  2.3624e-02, -9.6928e-01, -2.1931e-01],
                             [ 9.6956e-01,  5.8579e-03, -2.4477e-01, -2.2846e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ["spot_indoor_building_loop", "spot_indoor_obstacles"]:
        return torch.tensor([[ 0.0025, -1.0000, -0.0020,  0.0599],
                             [ 0.0125,  0.0021, -0.9999, -0.1898],
                             [ 0.9999,  0.0025,  0.0125, -0.1584],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ['spot_outdoor_day_srt_under_bridge_1', 'spot_outdoor_day_srt_under_bridge_2']:
        return torch.tensor([[ 2.8731e-03, -9.9999e-01, -1.6354e-03,  5.9817e-02],
                             [-2.7785e-04,  1.6346e-03, -1.0000e+00, -1.8998e-01],
                             [ 1.0000e+00,  2.8735e-03, -2.7315e-04, -1.5884e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([[ 0.0045, -0.9996, -0.0265,  0.0488],
                             [-0.2653,  0.0243, -0.9638, -0.2194],
                             [ 0.9641,  0.0114, -0.2651, -0.2299],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        return torch.tensor([[-1.0447e-04, -9.9970e-01, -2.4339e-02,  5.0678e-02],
                             [-2.4484e-01,  2.3624e-02, -9.6928e-01, -2.1931e-01],
                             [ 9.6956e-01,  5.8579e-03, -2.4477e-01, -2.2846e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['car_urban_day_penno_small_loop']:
        return torch.tensor([[ 4.3559e-03, -9.9987e-01, -1.5325e-02,  5.9558e-02],
                             [ 3.5460e-04,  1.5326e-02, -9.9988e-01, -1.8978e-01],
                             [ 9.9999e-01,  4.3499e-03,  4.2131e-04, -1.5842e-01],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['car_urban_night_penno_small_loop']:
        return torch.tensor([[ 0.0027, -0.9999, -0.0166,  0.0594],
                             [ 0.0027,  0.0166, -0.9999, -0.1903],
                             [ 1.0000,  0.0027,  0.0028, -0.1577],
                             [ 0.0000,  0.0000,  0.0000,  1.0000]])
    else:
        raise TypeError("Sequence Not Available")

def load_map(map_file, device):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to(device)
    return voxelized

def crop_local_map(PC_map, pose, velo2cam, max_depth):
    local_map = PC_map.clone()
    local_map = torch.mm(pose, local_map)
    indexes = local_map[0, :] > -1.
    if max_depth == 10.:
        indexes = indexes & (local_map[0, :] < 10.)
        indexes = indexes & (local_map[1, :] > -5.)
        indexes = indexes & (local_map[1, :] < 5.)
    elif max_depth == 100.:
        indexes = indexes & (local_map[0, :] < 100.)
        indexes = indexes & (local_map[1, :] > -25.)
        indexes = indexes & (local_map[1, :] < 25.)
    local_map = local_map[:, indexes]

    local_map = torch.mm(velo2cam, local_map)

    return local_map

def main(args):
    print(args)
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")

    calib = get_calib_m3ed(args.test_sequence)
    calib = calib.to(device)
    
    maps_file = os.path.join(args.data_folder, args.test_sequence, args.test_sequence+"_global.pcd")
    if os.path.exists(maps_file):
        print(f'load pointclouds from {maps_file}')
        vox_map = load_map(maps_file, device)
        print(f'load pointclouds finished! {vox_map.shape[1]} points')
    else:
        print("LiDAR map does not exist")
        sys.exit()

    velo2cam = get_velo2cam_m3ed(args.test_sequence)
    velo2cam = velo2cam.float().to(device)

    # load GT poses
    print('load ground truth poses')
    pose_path = os.path.join(args.data_folder, args.test_sequence, args.test_sequence+"_pose_gt.h5")
    poses = h5py.File(pose_path,'r')
    Ln_T_L0 = poses['Ln_T_L0']
    Ln_T_L0 = torch.tensor(Ln_T_L0)
    print(len(Ln_T_L0))

    model = torch.nn.DataParallel(Backbone_Event_Offset_RT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)
    model.eval()

    if args.save_log:
        log_file = f'./logs/Ours_M3ED_{args.test_sequence}.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
        log_file.writerow(header)
    
    est_rot = []
    est_trans = []
    err_t_list = []
    err_r_list = []
    print('Start tracking using EVLoc...')
    # k = 71  #falcon_indoor_flight_3
    k = 130
    initial_T = Ln_T_L0[k+1][:3, 3]
    initial_R = quaternion_from_matrix(Ln_T_L0[k+1])
    est_rot.append(initial_R.to(device))
    est_trans.append(initial_T.to(device))
    data_generate = Data_preprocess([calib], args.occlusion_threshold, args.occlusion_kernel)
    end = time.time()
    # for idx in range(len(Ln_T_L0)-1):
    for idx in range(k, len(Ln_T_L0)-70):
        initial_R = est_rot[idx-k].to(device)
        initial_T = est_trans[idx-k].to(device)

        RT = to_rotation_matrix(initial_R, initial_T)
        RT = RT.to(device)

        event_frame = np.load(os.path.join(args.data_folder, args.test_sequence, "event_frames_"+args.method, "left", f"event_frame_{idx:05d}.npy"))
        event_frame = torch.tensor(event_frame).permute(2, 0, 1)
        event_frame[event_frame<0] = 0
        event_frame /= torch.max(event_frame)       

        local_map = crop_local_map(vox_map, RT, velo2cam, max_depth=args.max_depth)   # 4xN
        event_input, lidar_input, _, _ = data_generate.push_input([event_frame], [local_map], None, None, device, MAX_DEPTH=args.max_depth, split='test')

        # flag = True
        flag = False
        if flag:
            original_overlay = overlay_imgs(event_frame[0, :, :, :], sparse[0, 0, :, :].clone()*0)
            cv2.imwrite(f"./visualization/{idx:05d}_img.png", original_overlay)
            original_overlay = overlay_imgs(event_frame[0, :, :, :]*0, sparse[0, 0, :, :].clone())
            cv2.imwrite(f"./visualization/{idx:05d}_depth.png", original_overlay)
            cv2.imwrite(f"./visualization/{idx:05d}_depth_.png", (sparse[0, 0, :, :].cpu().numpy()*255).astype(np.uint8))
            sys.exit()

        # run model
        _, flow_up, offset_R, offset_T = model(lidar_input, event_input, iters=24, test_mode=True)

        # update current pose
        R_pred, T_pred, _, _ = Flow2Pose(flow_up, lidar_input, [calib], MAX_DEPTH=args.max_depth, flag=False)     
        RT_pred = to_rotation_matrix(R_pred, T_pred)
        RT_pred = RT_pred.to(device)

        R_offset = offset_R[0]
        T_offset = offset_T[0]
        RT_offset = to_rotation_matrix(R_offset, T_offset)
        RT_pred_offset = torch.mm(RT_pred, RT_offset.inverse().to(RT_pred.device))

        T_pred_offset = RT_pred_offset[:3, 3]
        R_pred_offset = quaternion_from_matrix(RT_pred_offset)

        # RT_new = torch.mm(RT, RT_pred_offset)
        RT_new = torch.mm(velo2cam.inverse(), torch.mm(RT_pred_offset, torch.mm(velo2cam, RT)))

        # calculate RTE RRE
        predicted_R = quaternion_from_matrix(RT_new)
        predicted_T = RT_new[:3, 3]
        GT_T = Ln_T_L0[idx+1][:3, 3]
        GT_R = quaternion_from_matrix(Ln_T_L0[idx+1])
        err_r = quaternion_distance(predicted_R.unsqueeze(0).to(device),
                                    GT_R.unsqueeze(0).to(device), device=device)
        err_r = err_r * 180. / math.pi
        err_t = torch.norm(predicted_T.to(device) - GT_T.to(device)) * 100.
        err_r_list.append(err_r.item())
        err_t_list.append(err_t.item())

        print(f"{idx:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.std(err_t_list):.5f} "
              f"{np.std(err_r_list):.5f} {(time.time()-end)/(idx+1):.5f}")

        # update pose list
        est_rot[idx-k] = predicted_R
        est_trans[idx-k] = predicted_T
        est_rot.append(predicted_R)
        est_trans.append(predicted_T)
        # est_rot.append(GT_R)
        # est_trans.append(GT_T)

        if args.save_log:
            predicted_T = predicted_T.cpu().numpy()
            predicted_R = predicted_R.cpu().numpy()

            log_string = [f"{idx:05d}", str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0])]
            log_file.writerow(log_string)

        if args.render:
            # overlay_vis = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :])
            # cv2.imwrite(f"./visualization/odometry/0_{idx:05d}.png", overlay_vis)
            _, lidar_input, _, _ = data_generate.push_input([event_frame], [local_map], [T_pred_offset], [R_pred_offset], device, MAX_DEPTH=args.max_depth, split='test')
            # event_frame = warp(event_frame.to(flow_up.device), flow_up.detach())
            # original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :])
            # cv2.imwrite(f"./visualization/odometry/1_{idx:05d}.png", original_overlay)
            event_frame_vis = torch.sum(event_input[0, ...], dim=0)
            event_frame_vis = event_frame_vis.cpu().detach().numpy()
            event_frame_vis = (event_frame_vis / np.max(event_frame_vis) * 255).astype(np.uint8)
            lidar_input[lidar_input == 0] = 1000. 
            lidar_input = -lidar_input[0, 0, ...]
            lidar_input = lidar_input.clone()
            lidar_input = lidar_input.unsqueeze(0)
            lidar_input = lidar_input.unsqueeze(0)
            lidar_input = F.max_pool2d(lidar_input, 3, 1, 1)
            lidar_input = -lidar_input
            lidar_input[lidar_input == 1000.] = 0.
            lidar_input = lidar_input[0][0]
            lidar_frame_vis = lidar_input.cpu().detach().numpy()
            lidar_frame_vis = (lidar_frame_vis / np.max(lidar_frame_vis) * 255).astype(np.uint8)
            overlay_vis = np.zeros([event_frame_vis.shape[0], event_frame_vis.shape[1], 3])
            overlay_vis[:, :, 0] = event_frame_vis
            overlay_vis[:, :, 2] = lidar_frame_vis
            cv2.imwrite(f"./visualization/odometry/{idx:05d}.png", overlay_vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, metavar='DIR',
                        default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED',
                        help='path to dataset')
    parser.add_argument('--test_sequence', type=str, default="falcon_indoor_flight_3")
    parser.add_argument('--method', type=str, default='ours_denoise_pre_100000')
    parser.add_argument('--occlusion_kernel', type=float, default=5.)
    parser.add_argument('--occlusion_threshold', type=float, default=3.)
    parser.add_argument('--load_checkpoints', type=str, 
                        default="/home/eason/WorkSpace/EventbasedVisualLocalization/E2D_Loc/checkpoints/paper/2024-08-09-10-01-51/best_model.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--max_depth', type=float, default=10.)
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    main(args)