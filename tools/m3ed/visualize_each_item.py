import os
import glob
import h5py
import argparse
import numpy as np
import cv2
import torch
from utils import depth_generation
import torch.nn.functional as F
from matplotlib import cm


def overlay_imgs(rgb, lidar):
    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    if rgb.shape[2] == 3:
        rgb = rgb
    else:
        rgb = np.concatenate((np.zeros([rgb.shape[0], rgb.shape[1], 1]), rgb), axis=2)

    lidar[lidar == 0] = 1000. 
    lidar = -lidar

    lidar = lidar.clone()
    lidar = lidar.unsqueeze(0)
    lidar = lidar.unsqueeze(0)

    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    lidar = lidar[0][0]


    lidar = lidar.cpu().numpy()
    min_d = 0
    max_d = np.max(lidar)
    lidar = ((lidar - min_d) / (max_d - min_d)) * 255
    lidar = lidar.astype(np.uint8)
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) \
                + rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
    blended_img = (blended_img*255).astype(np.uint8)
    return blended_img

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

def get_left_right_T(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 9.9999e-01, -1.2158e-04, -4.6864e-03, -1.2023e-01],
                             [ 1.2021e-04,  1.0000e+00, -2.9335e-04,  8.3630e-04],
                             [ 4.6865e-03,  2.9279e-04,  9.9999e-01,  1.0119e-03],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[ 9.9999e-01, -6.6613e-04, -3.5103e-03, -1.2018e-01],
                             [ 6.6661e-04,  1.0000e+00,  1.3561e-04,  9.1033e-04],
                             [ 3.5102e-03, -1.3795e-04,  9.9999e-01, -4.3059e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ["spot_indoor_building_loop", "spot_indoor_obstacles"]:
        return torch.tensor([[ 9.9999e-01, -5.4043e-04, -3.8851e-03, -1.2024e-01],
                             [ 5.3475e-04,  1.0000e+00, -1.4630e-03,  9.3071e-04],
                             [ 3.8859e-03,  1.4609e-03,  9.9999e-01, -3.1356e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['spot_outdoor_day_srt_under_bridge_1']:
        return torch.tensor([[ 9.9999e-01, -6.4771e-04, -3.6507e-03, -1.2011e-01],
                             [ 6.3675e-04,  1.0000e+00, -3.0024e-03,  8.7946e-04],
                             [ 3.6526e-03,  3.0000e-03,  9.9999e-01, -2.8266e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_day_penno_parking_1', 'falcon_outdoor_day_penno_parking_2']:
        return torch.tensor([[ 9.9999e-01, -3.8986e-04, -3.2093e-03, -1.2005e-01],
                             [ 3.8672e-04,  1.0000e+00, -9.8024e-04,  9.3651e-04],
                             [ 3.2097e-03,  9.7899e-04,  9.9999e-01,  3.4862e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['falcon_outdoor_night_penno_parking_1', 'falcon_outdoor_night_penno_parking_2']:
        return torch.tensor([[ 9.9999e-01, -6.6613e-04, -3.5103e-03, -1.2018e-01],
                             [ 6.6661e-04,  1.0000e+00,  1.3561e-04,  9.1033e-04],
                             [ 3.5102e-03, -1.3795e-04,  9.9999e-01, -4.3059e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['car_urban_day_penno_small_loop']:
        return torch.tensor([[ 9.9999e-01, -7.5328e-04, -3.4030e-03, -1.2022e-01],
                             [ 7.4723e-04,  1.0000e+00, -1.7777e-03,  9.0584e-04],
                             [ 3.4043e-03,  1.7752e-03,  9.9999e-01, -1.1366e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    elif sequence in ['car_urban_night_penno_small_loop']:
        return torch.tensor([[ 1.0000e+00, -6.0816e-04, -2.7575e-03, -1.2019e-01],
                             [ 6.0945e-04,  1.0000e+00,  4.6747e-04,  1.0605e-03],
                             [ 2.7572e-03, -4.6915e-04,  1.0000e+00,  1.0729e-04],
                             [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    else:
        raise TypeError("Sequence Not Available")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="spot_indoor_building_loop",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--method",
                    default="ours_denoise_pre_100000",
                    help="Event representation method",
                    type=str)
    ap.add_argument("--camera",
                    default="left",
                    help="which camera to use",
                    type=str)    
    args = ap.parse_args()    

    data_path = os.path.join(args.dataset, args.sequence)

    event_paths = os.path.join(data_path, f"event_frames_{args.method}", args.camera)
    pc_paths = os.path.join(data_path, 'local_maps')
    
    save_path = os.path.join("./visualization", args.method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sequences = sorted(glob.glob(f"{event_paths}/*.npy"))
    for idx in range(len(sequences)):
        event_path = sequences[idx]
        print(idx, event_path)
        sequence = event_path.split('/')[-1].split('_')[-1].split('.')[0]
        pc_path = os.path.join(pc_paths, f'point_cloud_{int(sequence)+1:05d}.h5')

        events = np.load(event_path)
        event_time_image = events
        event_time_image[event_time_image<0]=0
        event_time_image = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
        event_time_image = (event_time_image / np.max(event_time_image) * 255).astype(np.uint8)
        event_time_image = event_time_image[60:660, 160:960+160, :]
        # event_time_image = event_time_image[32:32+296, 64:64+512, :]
        cv2.imwrite(f"{save_path}/{idx:05d}_event_frame.png", event_time_image)
        
        try:
            with h5py.File(pc_path, 'r') as hf:
                pc = hf['PC'][:]
        except Exception as e:
            print(f'File Broken: {pc_path}')
            raise e
        pc_in = torch.from_numpy(pc.astype(np.float32))
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.

        if args.camera == 'right':
            T_to_prophesee_left = get_left_right_T(args.sequence)
            pc_in = torch.matmul(T_to_prophesee_left, pc_in)

        calib = get_calib_m3ed(args.sequence)
        calib = calib.cuda().float()
        sparse_depth = depth_generation(pc_in.cuda(), (720, 1280), calib, 3., 5, device='cuda:0')
        # sparse_depth = depth_generation(pc_in.cuda(), (360, 640), calib / 2., 3., 5, device='cuda:0')
        sparse_depth = overlay_imgs(sparse_depth.repeat(3, 1, 1)*0, sparse_depth[0, ...])
        sparse_depth = sparse_depth[60:660, 160:960+160, :]
        # sparse_depth = sparse_depth[32:32+296, 64:64+512, :]
        cv2.imwrite(f"{save_path}/{idx:05d}_depth.png", sparse_depth)


