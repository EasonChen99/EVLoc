import os
import sys
import h5py
import numpy as np
import cv2
import torch
import mathutils
from tqdm import tqdm
from core.utils_point import invert_pose, overlay_imgs
from core.utils_events import load_data, load_map, find_near_index
from core.data_preprocess import Data_preprocess
import visibility
from core.flow_viz import flow_to_image

import argparse
from core.backbone import Backbone
from core.flow2pose import Flow2Pose, err_Pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoints',
                        help="restore checkpoint")
    parser.add_argument('--wdecay', 
                        type=float, 
                        default=.00005)
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=1e-8)
    parser.add_argument('--clip', 
                        type=float, 
                        default=1.0)
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.8, 
                        help='exponential weighting')
    parser.add_argument('--max_r', 
                        type=float, 
                        default=5.)
    parser.add_argument('--max_t', 
                        type=float, 
                        default=0.5)
    parser.add_argument('--time_window', 
                        type=int, 
                        default=100000)  
    parser.add_argument('--idx', 
                        type=int, 
                        default=0)       
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=3)
    parser.add_argument('--mixed_precision', 
                        action='store_true', 
                        help='use mixed precision')
    parser.add_argument('--use_uncertainty', 
                        action='store_true')
    parser.add_argument('--use_offset_learning',
                        action='store_true',
                        help='estimate offsets')
    parser.add_argument('--render', 
                        action='store_true')
    parser.add_argument('--render_event', 
                        action='store_true')    
    args = parser.parse_args()  



    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    name = "falcon_indoor_flight_3"
    root = "/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/M3ED/Falcon/Indoor/flight_3"
    h5_file = f"{name}_data.h5"
    pose_file = f"{name}_depth_gt.h5"
    pc_file = f"{name}_global.pcd"

    data_path = os.path.join(root, h5_file)
    pose_path = os.path.join(root, pose_file)
    pc_path = os.path.join(root, pc_file)

    data = h5py.File(data_path,'r')
    rgb_data = load_data(data, sensor='ovc', camera='rgb') 
    event_data = load_data(data, sensor='prophesee', camera='left')
    # # pose load
    poses = h5py.File(pose_path,'r')
    depths = poses['depth/prophesee/left']
    Cn_T_C0 = poses['Cn_T_C0']                                                  # 570 4 4
    Ln_T_L0 = poses['Ln_T_L0']                                                  # 570 4 4
    pose_ts = poses['ts']                                                       # 570
    ts_map_prophesee_left = poses['ts_map_prophesee_left']                      # 570 index

    # initialize pose error
    max_angle = args.max_r
    max_t = args.max_t
    rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-min(max_t, 1.), min(max_t, 1.))
    R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
    T = mathutils.Vector((transl_x, transl_y, transl_z))
    R, T = invert_pose(R, T)
    R, T = torch.tensor(R), torch.tensor(T)
    # R = torch.tensor([ 9.9932e-01, -1.3382e-02, -3.4292e-02, -1.9004e-04])
    # T = torch.tensor([-0.0512,  0.3071,  0.4393])

    i = args.idx
    time_window = args.time_window

    # time_windows = [50, 100, 200, 300]
    # for i in tqdm(range(275, 489)):
    #     for j in range(0, 4):
    #         time_window = time_windows[j] * 1000

    idx_cur = int(ts_map_prophesee_left[i+1])
    t_ref = event_data['t'][idx_cur]
    idx_start, idx_cur, idx_end = find_near_index(event_data['t'][idx_cur], event_data['t'], time_window=time_window)
    if args.render_event:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    rows, cols = event_data['resolution'][1], event_data['resolution'][0]
    event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
    for j in tqdm(range(idx_end-idx_start)):
        idx = int(j + idx_start)
        y, x = event_data['y'][idx], event_data['x'][idx]
        if event_data['p'][idx] > 0:
            event_time_image[y, x, 0] = event_data['t'][idx]
        else:
            event_time_image[y, x, 1] = event_data['t'][idx]
        
        if args.render_event:
            coordinate = [j / (idx_end-idx_start) * 1000, x, y]
            if event_data['p'][idx] > 0:
                ax.scatter(*coordinate, color='r')
            else:
                ax.scatter(*coordinate, color='b')
    
    event_time_image[event_time_image > 0] -= event_data['t'][idx_start]
    event_time_image = torch.tensor(event_time_image, device=device)

    if args.render_event:
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        # Setting the limits for clarity
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_zlim([0, 1000])
        # Show the plot
        plt.savefig("./visualization/events.png")
    
    # # pc map load
    vox_map = load_map(pc_path, device)                                         # 4 N
    print(f'load pointclouds finished! {vox_map.shape[1]} points')
    pose = torch.tensor(Ln_T_L0[i+1], device=device, dtype=torch.float32)
    pose_inv = pose.inverse()
    local_map = vox_map.clone()
    local_map = torch.matmul(pose, local_map)
    indexes = local_map[0, :] > -1.
    indexes = indexes & (local_map[0, :] < 10.)
    indexes = indexes & (local_map[1, :] > -5.)
    indexes = indexes & (local_map[1, :] < 5.)
    local_map = local_map[:, indexes]
    prophesee_left_T_lidar = torch.tensor(data["/ouster/calib/T_to_prophesee_left"], device=device, dtype=torch.float32)
    local_map = torch.matmul(prophesee_left_T_lidar, local_map)
    # pc_visualize(local_map.t().cpu().numpy())
    calib = torch.tensor(event_data['intrinsics'], device=device, dtype=torch.float)
    data_generate = Data_preprocess(calib.unsqueeze(0), 3, 5)
    event_input, lidar_input, flow_gt = data_generate.push(event_time_image.permute(2, 0, 1).unsqueeze(0), 
                                                        local_map.unsqueeze(0), 
                                                        T.unsqueeze(0), R.unsqueeze(0), 
                                                        device, split='test')  


    model = torch.nn.DataParallel(Backbone(args), device_ids=[0])
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)
    model.eval()

    _, flow_up, _, _ = model(lidar_input, event_input, iters=24, test_mode=True)

    epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    val = valid_gt.view(-1) >= 0.5
    if val.sum() == 0:
        val = lidar_input.view(-1) > 0

    R_pred, T_pred, inliers, _ = Flow2Pose(flow_up, lidar_input, calib.unsqueeze(0))
    R_pred_2, T_pred_2, _, flag = Flow2Pose(flow_up, lidar_input, calib.unsqueeze(0), flow_gt=flow_gt)
    err_r, err_t = err_Pose(R_pred, T_pred, R, T)
    print(f"{epe[val].mean().item()} {err_r.item():.5f} {err_t.item():.5f}")
    if not flag:
        err_r_2, err_t_2 = err_Pose(R_pred_2, T_pred_2, R, T)
        print(f"{epe[val].mean().item()} {err_r_2.item():.5f} {err_t_2.item():.5f}")


    ## render
    if args.render:
        vis_event_time_image = event_input[0, ...].permute(1, 2, 0).cpu().numpy()
        vis_event_time_image = np.concatenate((np.zeros([vis_event_time_image.shape[0], vis_event_time_image.shape[1], 1]), vis_event_time_image), axis=2)
        vis_event_time_image = (vis_event_time_image / np.max(vis_event_time_image) * 255).astype(np.uint8)
        cv2.imwrite(f"./visualization/{i:03d}_event_{time_window//1000}.png", vis_event_time_image)

        flow_image = flow_to_image(flow_up.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
        cv2.imwrite("./visualization/flow.png", flow_image)

        output = torch.zeros(flow_up.shape).to(device)
        pred_depth_img = torch.zeros(lidar_input.shape).to(device)
        pred_depth_img += 1000.
        output = visibility.image_warp_index(lidar_input.to(device),
                                            flow_up.int().to(device), pred_depth_img,
                                            output, lidar_input.shape[3], lidar_input.shape[2])
        pred_depth_img[pred_depth_img == 1000.] = 0.

        original_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input[0, 0, :, :])
        cv2.imwrite(f"./visualization/{i:03d}_depth_ori_{time_window//1000}.png", original_overlay)
        warp_overlay = overlay_imgs(event_input[0, :, :, :]*0, pred_depth_img[0, 0, :, :])
        
        _, lidar_input_pred, _ = data_generate.push(event_time_image.permute(2, 0, 1).unsqueeze(0), 
                                                    local_map.unsqueeze(0), 
                                                    T_pred.unsqueeze(0), R_pred.unsqueeze(0), 
                                                    device, split='test') 
        pred_overlay = overlay_imgs(event_input[0, :, :, :]*0, lidar_input_pred[0, 0, :, :])
        cv2.imwrite(f"./visualization/{i:03d}_depth_prediction_{time_window//1000}.png", pred_overlay)
        
        inliers_map = np.zeros([lidar_input.shape[2], lidar_input.shape[3], 3])
        inliers_map[inliers[:, 0], inliers[:, 1], 0] = 255

        cat_ori = np.hstack((vis_event_time_image, original_overlay))
        original_overlay = overlay_imgs(event_input[0, :, :, :], lidar_input[0, 0, :, :])
        cat_pre = np.hstack((original_overlay, flow_image))
        cat = np.vstack((cat_ori, cat_pre))

        cat_right = np.vstack((warp_overlay, inliers_map))
        cat = np.hstack((cat, cat_right))
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        ##
        text = f"idx={i} window={time_window // 1000}"
        org = (cat.shape[1]-950, cat.shape[0]-100)
        cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        ##
        initial_err_r, initial_err_t = err_Pose(torch.tensor([1., 0., 0., 0.], device=R.device), torch.tensor([0., 0., 0.], device=R.device), R, T)
        text = f"R={initial_err_r.item():.3f} T={initial_err_t.item():.3f}"
        org = (cat.shape[1]-950, cat.shape[0]-70)
        cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        ##
        text = f"R={err_r.item():.3f} T={err_t.item():.3f} epe={epe[val].mean().item():.3f} epe_inliers={epe[(inliers_map[:, :, 0].flatten()>0)].mean().item():.3f}"
        org = (cat.shape[1]-950, cat.shape[0]-40)
        cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        ##
        text = f"inliers={inliers_map.sum():.0f} epe_less_5={(epe[val] < 5).sum():.0f}"
        org = (cat.shape[1]-950, cat.shape[0]-10)
        cv2.putText(cat, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

                # cv2.imwrite(f'./visualization/cat.png', cat)
                # cv2.imwrite(f'/home/eason/WorkSpace/EventbasedVisualLocalization/E2D_Loc/visualization/analysis/0_00_output_single/change_window_length/{i:03d}_{time_window}.png', cat)