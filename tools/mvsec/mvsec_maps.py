import numpy as np
import os
import h5py
import argparse
import tqdm
import open3d as o3
from utils import load_map
import torch

np.set_printoptions(threshold=np.inf)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--save_dir', type=str, default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/MVSEC', metavar='PARAMS', help='Main Directory to save all encoding results')
    parser.add_argument('--save_env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--data_path', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/MVSEC', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
    parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
    parser.add_argument('--map2pc', action='store_true')
    args = parser.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.data_path, args.save_env+"_data.hdf5")
    d_set = h5py.File(data_path, 'r')
    gray_images_ts = d_set['davis']['left']['image_raw_ts']
    lidar_scans_data = d_set['velodyne']['scans']
    lidar_scans_ts = d_set['velodyne']['scans_ts']
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

    print(gray_images_ts.shape, lidar_scans_ts.shape, poses_ts.shape)
    print(gray_images_ts[:10]/1e5)
    print(lidar_scans_ts[:10]/1e5)
    print(poses_ts[:10]/1e5)

    if not args.map2pc:
        pcl = o3.geometry.PointCloud()
        for i in tqdm.tqdm(range(poses.shape[0])):
            scan_idx = i + 5
            
            lidar_scan = lidar_scans_data[scan_idx, ...]
            pc = lidar_scan[:, :3]
            intensity = lidar_scan[:, 3]
            RT = np.matmul(poses[i, ...], T_cam0_lidar)

            valid_indices = pc[:, 0] < -3.
            valid_indices = valid_indices | (pc[:, 0] > 3.)
            valid_indices = valid_indices | (pc[:, 1] < -3.)
            valid_indices = valid_indices | (pc[:, 1] > 3.)
            pc = pc[valid_indices].copy()
            pc = np.concatenate((pc, np.ones([pc.shape[0], 1])), axis=1)

            pc_rot = np.matmul(RT, pc.T)
            pc_rot = pc_rot.astype(np.float).T.copy()        

            pcl_local = o3.geometry.PointCloud()
            pcl_local.points = o3.utility.Vector3dVector(pc_rot[:, :3])
            pcl_local.colors = o3.utility.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)

            # downpcd = pcl_local.voxel_down_sample(voxel_size=args.voxel_size)
            downpcd = pcl_local

            pcl.points.extend(downpcd.points)
            pcl.colors.extend(downpcd.colors)
        
        # downpcd_full = pcl.voxel_down_sample(voxel_size=args.voxel_size)
        downpcd_full = pcl
        downpcd, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)
        o3.io.write_point_cloud(os.path.join(args.save_dir, args.save_env, f'mvsec_{args.save_env}_global.pcd'), downpcd)
    else:
        map_file = os.path.join(args.save_dir, args.save_env, f'mvsec_{args.save_env}_global.pcd')
        if not os.path.exists:
            raise "Map file doesn't exit."
        vox_map = load_map(map_file, device)

        out_path = os.path.join(args.save_dir, args.save_env, "local_maps")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in tqdm.tqdm(range(poses.shape[0])):
            file_path = os.path.join(out_path, f'point_cloud_{i:05d}.h5')
            if os.path.exists(file_path):
                continue
            pose = torch.tensor(poses[i, ...], device=device, dtype=torch.float32)

            local_map = vox_map.clone()
            local_map = torch.matmul(pose.inverse(), local_map)
            indexes = local_map[2, :] > -1.

            indexes = indexes & (local_map[2, :] < 10.)
            indexes = indexes & (local_map[1, :] > -5.)
            indexes = indexes & (local_map[1, :] < 5.)
            # indexes = indexes & (local_map[0, :] < 10.)
            # indexes = indexes & (local_map[1, :] > -10.)
            # indexes = indexes & (local_map[1, :] < 10.)

            local_map = local_map[:, indexes]

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)