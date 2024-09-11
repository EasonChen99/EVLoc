import os
import h5py
import open3d as o3
import torch
import argparse
from utils import load_map
import tqdm 

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
    ap.add_argument("--save_dir",
                    default="/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED",
                    help="Path to save preprocessed data",
                    type=str)
    args = ap.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.dataset, args.sequence+"_data.h5")
    pose_path = os.path.join(args.dataset, args.sequence+"_pose_gt.h5")
    pc_path = os.path.join(args.dataset, args.sequence+"_global.pcd")

    out_path = os.path.join(args.save_dir, args.sequence, "local_maps")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    data = h5py.File(data_path,'r')
    prophesee_left_T_lidar = torch.tensor(data["/ouster/calib/T_to_prophesee_left"], device=device, dtype=torch.float32)
    # print(prophesee_left_T_lidar) 

    # # pose load
    poses = h5py.File(pose_path,'r')
    Cn_T_C0 = poses['Cn_T_C0']                                                  # 570 4 4
    Ln_T_L0 = poses['Ln_T_L0']                                                  # 570 4 4
    pose_ts = poses['ts']                                                       # 570
    ts_map_prophesee_left = poses['ts_map_prophesee_left']                      # 570 index

    # # pc map load
    vox_map = load_map(pc_path, device)                                         # 4 N
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    for idx in tqdm.tqdm(range(Ln_T_L0.shape[0]-1)):
        file = os.path.join(out_path, f'point_cloud_{idx:05d}.h5')
        if os.path.exists(file):
            continue
        pose = torch.tensor(Ln_T_L0[idx], device=device, dtype=torch.float32)
        local_map = vox_map.clone()
        local_map = torch.matmul(pose, local_map)
        indexes = local_map[0, :] > -1.
        
        ## falcon_indoor_flight
        # indexes = indexes & (local_map[0, :] < 10.)
        # indexes = indexes & (local_map[1, :] > -5.)
        # indexes = indexes & (local_map[1, :] < 5.)

        ## falcon_outdoor_day_penno_parking
        # indexes = indexes & (local_map[0, :] < 100.)
        # indexes = indexes & (local_map[1, :] > -25.)
        # indexes = indexes & (local_map[1, :] < 25.)
        
        ## spot_outdoor_day_srt_under_bridge
        indexes = indexes & (local_map[0, :] < 50.)
        indexes = indexes & (local_map[1, :] > -15.)
        indexes = indexes & (local_map[1, :] < 15.)
        
        local_map = local_map[:, indexes]
        local_map = torch.matmul(prophesee_left_T_lidar, local_map)

        with h5py.File(file, 'w') as hf:
            hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)