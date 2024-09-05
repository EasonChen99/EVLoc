import numpy as np
import os
import h5py
import argparse
import tqdm
import open3d as o3
from utils import load_map
import torch
from utils_point import to_rotation_matrix

np.set_printoptions(threshold=np.inf)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--save-dir', type=str, default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/VECtor', metavar='PARAMS', help='Main Directory to save all encoding results')
    parser.add_argument('--save-env', type=str, default='corridors_dolly1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--data-path', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/VECtor', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
    parser.add_argument('--voxel_size', default=0.1, type=float, help='Voxel Size')
    parser.add_argument('--map2pc', action='store_true')
    args = parser.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    data_handler = open(os.path.join(args.data_path, args.save_env, args.save_env+".synced.gt.txt"), 'r')
    all_data = data_handler.readlines()
    len_pose = len(all_data) - 2
    poses = []
    for idx in range(len(all_data)):
        if all_data[idx][0] == '#':
            continue
        line = all_data[idx].split(" ")
        pose = []
        for j in range(8):
            pose.append(float(line[j]))
        poses.append(pose)

    # transformation from LiDAR to left regular camera
    T_cam2_lidar = np.array(
                    [[ 0.0119197 , -0.999929  ,  0.0000523,  0.0853154],
                     [-0.00648951, -0.00012969, -0.999979 , -0.0684439],
                     [0.999908  ,  0.0119191 , -0.0064906, -0.0958121],
                     [0         ,  0         ,  0        ,  1        ]]  
                    )
    # transformation from left event camera to left regular camera
    T_cam0_cam2 = np.array(
                [[ 0.9998732356434525 , 0.01166113698213495, -0.01084114976267556, -0.0007543180009142757], 
                 [-0.01183095928451621, 0.9998062047518974 , -0.01573471772912168, -0.04067615384902421  ], 
                 [0.01065556410055307, 0.01586098432919285,  0.9998174273985267 , -0.01466127320771003  ],
                 [0                  , 0                  ,  0                  ,  1                    ]]  
                )
    # transformation from LiDAR to left event camera
    T_cam0_lidar = np.matmul(np.linalg.inv(T_cam0_cam2), T_cam2_lidar)

    if not args.map2pc:
        pcl = o3.geometry.PointCloud()
        for i in tqdm.tqdm(range(len(poses))):
            pose = poses[i]
            T = torch.tensor([pose[1], pose[2], pose[3]])
            R = torch.tensor([pose[4], pose[5], pose[6], pose[7]])
            RT = to_rotation_matrix(R, T)
            RT = RT.numpy().astype(np.float16)

            # lidar_scan = lidar_scans_data[scan_idx, ...]
            lidar_scan = o3.io.read_point_cloud(os.path.join(args.data_path, args.save_env, args.save_env+".synced.lidar", f"{pose[0]:.6f}.pcd"))
            pc = np.array(lidar_scan.points, dtype=np.float32)
            intensity = np.ones([1, pc.shape[0]])
            RT = np.matmul(RT, T_cam0_lidar)

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
        for i in tqdm.tqdm(range(len(poses))):
            file_path = os.path.join(out_path, f'point_cloud_{i:05d}.h5')
            if os.path.exists(file_path):
                continue
            # pose = torch.tensor(poses[i, ...], device=device, dtype=torch.float32)
            pose = poses[i]
            T = torch.tensor([pose[1], pose[2], pose[3]])
            R = torch.tensor([pose[4], pose[5], pose[6], pose[7]])
            pose = to_rotation_matrix(R, T)

            local_map = vox_map.clone()
            local_map = torch.matmul(pose.inverse().to(device), local_map)
            indexes = local_map[2, :] > -1.

            indexes = indexes & (local_map[2, :] < 10.)
            indexes = indexes & (local_map[1, :] > -5.)
            indexes = indexes & (local_map[1, :] < 5.)

            local_map = local_map[:, indexes]

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)