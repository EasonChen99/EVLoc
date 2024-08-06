import numpy as np
import open3d as o3
import torch
import math
import cv2
import sys
import bisect
import visibility
from core.camera_model import CameraModel

def load_data(data, sensor='ovc', camera='rgb'):
    '''
        load data from hdf5 file
    '''
    if sensor not in ['ovc', 'prophesee', 'ouster']:
        raise "The sensor doesn't exist."

    if sensor == 'ovc':
        if camera not in ['left', 'right', 'rgb']:
            raise "The camera does not exist."
        
        data = data['ovc']
        ts = data['ts']                                                     # 1424 (interval=40000us)
        ts_map_prophesee_left_t = data['ts_map_prophesee_left_t']           # 1424 index
        data = data[camera]
        img = data['data']                                                  # (1424 800 1280 3)
        calib = data['calib']
        T_to_prophesee_left = np.asarray(calib['T_to_prophesee_left'])      # 4 4
        distortion_coeffs = np.asarray(calib['distortion_coeffs'])          # 4
        intrinsics = np.asarray(calib['intrinsics'])                        # 4
        resolution = np.asarray(calib['resolution'])                        # (1424 800)
        sample = {"rgb": img, "ts": ts, "ts_map_prophesee_left_t": ts_map_prophesee_left_t,
                "T_to_prophesee_left": T_to_prophesee_left, "distortion_coeffs": distortion_coeffs,
                "intrinsics": intrinsics, "resolution": resolution}
    
    elif sensor == 'prophesee':
        if camera not in ['left', 'right']:
            raise "The camera does not exist."
        
        data = data['prophesee']
        # print(data['left']['t'][0], data['left']['t'][-1])
        # print(data['right']['t'][0], data['right']['t'][-1])
        data = data[camera]
        calib = data['calib']
        distortion_coeffs = np.asarray(calib['distortion_coeffs'])      # 4
        intrinsics = np.asarray(calib['intrinsics'])                    # 4
        resolution = np.asarray(calib['resolution'])                    # (1280 720)
        ms_map_idx = data['ms_map_idx'][:]
        p, t, x, y = data['p'], data['t'], data['x'], data['y']
        T_to_prophesee_left = np.asarray(calib['T_to_prophesee_left'])

        sample = {"p": p, "t": t, "x": x, "y": y,
                  "distortion_coeffs": distortion_coeffs, "intrinsics": intrinsics, 'T_to_prophesee_left': T_to_prophesee_left,
                  "resolution": resolution, "ms_map_idx": ms_map_idx}
    
    return sample

def find_near_index(t_cur, ts, time_window=100000):
    half_time_window = time_window // 2

    min_t = t_cur - half_time_window
    max_t = t_cur + half_time_window

    # Using bisect to find the closest index to t_cur
    pos_cur_t = bisect.bisect_left(ts, t_cur)
    if pos_cur_t == 0:
        idx_cur = pos_cur_t
    elif pos_cur_t == len(ts):
        idx_cur = pos_cur_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_cur = pos_cur_t if abs(ts[pos_cur_t] - min_t) < abs(ts[pos_cur_t - 1] - min_t) else pos_cur_t - 1

    # Using bisect to find the closest index to min_t
    pos_min_t = bisect.bisect_left(ts, min_t)
    if pos_min_t == 0:
        idx_start = pos_min_t
    elif pos_min_t == len(ts):
        idx_start = pos_min_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_start = pos_min_t if abs(ts[pos_min_t] - min_t) < abs(ts[pos_min_t - 1] - min_t) else pos_min_t - 1
    
    # Using bisect to find the closest index to max_t
    pos_max_t = bisect.bisect_left(ts, max_t)
    if pos_max_t == 0:
        idx_end = pos_max_t
    elif pos_max_t == len(ts):
        idx_end = pos_max_t - 1
    else:
        # Determine the closest index by comparing distances to the adjacent values
        idx_end = pos_max_t if abs(ts[pos_max_t] - max_t) < abs(ts[pos_max_t - 1] - max_t) else pos_max_t - 1


    return idx_start, idx_cur, idx_end


def load_map(map_file, device):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to(device)
    return voxelized

def pc_visualize(pointcloud):
    point_cloud = o3.open3d.geometry.PointCloud()
    point_cloud.points = o3.open3d.utility.Vector3dVector(pointcloud[:,0:3].reshape(-1,3))
    o3.open3d.visualization.draw_geometries([point_cloud],width=800,height=600)
    return point_cloud


def depth_generation(local_map, image_size, cam_params, occu_thre, occu_kernel, device, only_uv=False):
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _, refl, indexes = cam_model.project_withindex_pytorch(local_map, image_size)
    uv = uv.t().int().contiguous()
    if only_uv:
        return uv, indexes
    depth_img = torch.zeros(image_size[:2], device=device, dtype=torch.float)
    depth_img += 1000.
    idx_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    indexes = indexes.float()

    depth_img, idx_img = visibility.depth_image(uv, depth, indexes,
                                                depth_img, idx_img,
                                                uv.shape[0], image_size[1], image_size[0])
    depth_img[depth_img == 1000.] = 0.

    deoccl_index_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    projected_points = torch.zeros_like(depth_img, device=device)
    projected_points, _ = visibility.visibility2(depth_img, cam_params,
                                                 idx_img,
                                                 projected_points,
                                                 deoccl_index_img,
                                                 depth_img.shape[1],
                                                 depth_img.shape[0],
                                                 occu_thre,
                                                 int(occu_kernel))
    projected_points /= 10.
    projected_points = projected_points.unsqueeze(0)

    return projected_points