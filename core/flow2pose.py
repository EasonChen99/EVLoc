import torch
import visibility
import cv2
import numpy as np
import mathutils
import math

import sys
sys.path.append('core')
from camera_model import CameraModel
from utils_point import invert_pose, quat2mat, tvector2mat, quaternion_from_matrix, rotation_vector_to_euler
from quaternion_distances import quaternion_distance

import poselib

def Flow2Pose(flow_up, depth, calib, flow_gt=None, uncertainty=None):
    """
        flow_up: Bx2xHxW
        depth  : Bx1xHxW
        calib  : Bx4
    """
    device = flow_up.device

    if flow_gt is not None:
        epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
        valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
        mask = (epe < 5) * valid_gt # BxHxW
        # mask = valid_gt
        mask = mask[0, ...].cpu().numpy()
    if uncertainty is not None:
        mask = uncertainty < 0.7
        mask = mask[0, 0, ...].cpu().numpy()

    output = torch.zeros(flow_up.shape).to(device)
    pred_depth_img = torch.zeros(depth.shape).to(device)
    pred_depth_img += 1000.
    output = visibility.image_warp_index(depth.to(device), flow_up.int(), pred_depth_img, output,
                                         depth.shape[3], depth.shape[2])
    pred_depth_img[pred_depth_img == 1000.] = 0.
    pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()

    depth_img_ori = depth.cpu().numpy() * 10.

    mask_depth_1 = pc_project_uv[0, :, :, 0] != 0
    mask_depth_2 = pc_project_uv[0, :, :, 1] != 0
    mask_depth = mask_depth_1 + mask_depth_2
    depth_img = depth_img_ori[0, 0, :, :] * mask_depth

    if (flow_gt is not None) or (uncertainty is not None):
        depth_img = depth_img * mask

    cam_model = CameraModel()
    cam_params = calib[0].clone().cpu().numpy()
    x, y = 60, 160
    # x, y = 100, 160
    cam_params[2] = cam_params[2] + 480 - (y + y + 960) / 2.
    cam_params[3] = cam_params[3] + 300 - (x + x + 600) / 2.
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    cam_mat = np.array([[cam_params[0], 0, cam_params[2]], [0, cam_params[1], cam_params[3]], [0, 0, 1.]])

    pts3d, pts2d, indexes = cam_model.deproject_pytorch(depth_img, pc_project_uv[0, :, :, :])
    if pts3d.shape[0] < 4:
        return 0, 0, np.zeros([1, 2], dtype=np.uint8), True
    
    # _, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3d, pts2d, cam_mat, None)
    # # print("inlier rate:", inliers.shape[0] / pts3d.shape[0])
    # reuler = rotation_vector_to_euler(rvecs)
    # R = mathutils.Euler((reuler[0], reuler[1], reuler[2]), 'XYZ')
    # T = mathutils.Vector((tvecs[0], tvecs[1], tvecs[2]))
    # R, T = invert_pose(R, T)
    # R, T = torch.tensor(R), torch.tensor(T)
    # print(R, '\n', T)

    camera = {'model': 'SIMPLE_PINHOLE', 
              'width': 960, 'height': 600, 
              'params': [cam_params[0], cam_params[2], cam_params[3]]}
    pose, inliers = poselib.estimate_absolute_pose(pts2d, pts3d, camera, 
                                                {
                                                "max_reproj_error": 30.0,
                                                "seed": 0,
                                                "progressive_sampling" : False,
                                                "max_prosac_iterations": 50000,
                                                "real_focal_check": False,
                                                }, 
                                                {
                                                'loss_type': "HUBER",
                                                'loss_scale': 1.0,
                                                'gradient_tol': 1e-8,
                                                'step_tol': 1e-8,
                                                'initial_lambda': 1e-3,
                                                'min_lambda': 1e-10,
                                                'max_lambda': 1e10,
                                                'verbose': False,
                                                }
                                            )
    inliers = inliers['inliers']
    
    R = torch.tensor(pose.q)
    T = torch.tensor(pose.t)
    R[1:] *= -1
    T *= -1


    return R, T, indexes[inliers, :], False

def err_Pose(R_pred, T_pred, R_gt, T_gt):
    device = R_pred.device

    R = quat2mat(R_gt)
    T = tvector2mat(T_gt)
    RT_inv = torch.mm(T, R).to(device)
    RT = RT_inv.clone().inverse()

    R_pred = quat2mat(R_pred)
    T_pred = tvector2mat(T_pred)
    RT_pred = torch.mm(T_pred, R_pred)
    RT_pred = RT_pred.to(device)
    RT_new = torch.mm(RT, RT_pred)

    T_composed = RT_new[:3, 3]
    R_composed = quaternion_from_matrix(RT_new)
    R_composed = R_composed.unsqueeze(0)
    total_trasl_error = torch.tensor(0.0).to(device)
    total_rot_error = quaternion_distance(R_composed.to(device), torch.tensor([[1., 0., 0., 0.]]).to(device),
                                          device=R_composed.device)
    total_rot_error = total_rot_error * 180. / math.pi
    total_trasl_error += torch.norm(T_composed.to(device)) * 100.

    return total_rot_error, total_trasl_error


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask