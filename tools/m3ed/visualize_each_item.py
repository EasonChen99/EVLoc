import os
import glob
import h5py
import numpy as np
import cv2
import torch
from utils import depth_generation
from utils_point import overlay_imgs

root = f"/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/M3ED"

scene = "falcon_indoor_flight_1"
data_path = os.path.join(root, scene)

event_paths = os.path.join(data_path, "event_frames_ours_denoise_200000", 'left')
pc_paths = os.path.join(data_path, 'local_maps')

sequences = sorted(glob.glob(f"{event_paths}/*.npy"))
for idx in range(len(sequences)):
    # idx = 100
    # if idx < 70 or idx > len(sequences) - 70:
    #     continue
    event_path = sequences[idx]
    sequence = event_path.split('/')[-1].split('_')[-1].split('.')[0]
    pc_path = os.path.join(pc_paths, f'point_cloud_{int(sequence)+1:05d}.h5')

    events = np.load(event_path)
    event_time_image = events
    event_time_image[event_time_image<0]=0
    event_time_image = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
    event_time_image = (event_time_image / np.max(event_time_image) * 255).astype(np.uint8)
    cv2.imwrite(f"./visualization/{idx:05d}_event.png", event_time_image)
    
    event_time_image = cv2.medianBlur(event_time_image, 3)
    # event_time_image = cv2.GaussianBlur(event_time_image, (7,7), 0)
    event_time_image_denoise = event_time_image
    # event_time_image_denoise = cv2.bilateralFilter(event_time_image_denoise,9,255,255)
    cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_1.png", event_time_image_denoise)
    # # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (7,7), 0)
    # # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (5,5), 0)
    # event_time_image_denoise = cv2.morphologyEx(event_time_image_denoise, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_2.png", event_time_image_denoise)
    # edge = cv2.Canny(event_time_image_denoise, 255/3, 255)
    # cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_edge.png", edge)
    # event_time_image_denoise = cv2.medianBlur(event_time_image_denoise, 5)
    # cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_2.png", event_time_image_denoise)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    # event_time_image_denoise = cv2.morphologyEx(event_time_image_denoise, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_3.png", event_time_image_denoise)
    # event_time_image_denoise = event_time_image
    # event_time_image_denoise = cv2.medianBlur(event_time_image_denoise, 3)
    # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (7,7), 0)
    # event_time_image_denoise = cv2.GaussianBlur(event_time_image_denoise, (5,5), 0)
    # cv2.imwrite(f"./visualization/{idx:05d}_event_denoise_4.png", event_time_image_denoise)

    # try:
    #     with h5py.File(pc_path, 'r') as hf:
    #         pc = hf['PC'][:]
    # except Exception as e:
    #     print(f'File Broken: {pc_path}')
    #     raise e
    # pc_in = torch.from_numpy(pc.astype(np.float32))
    # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
    #     pc_in = pc_in.t()
    # if pc_in.shape[0] == 3:
    #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
    #     pc_in = torch.cat((pc_in, homogeneous), 0)
    # elif pc_in.shape[0] == 4:
    #     if not torch.all(pc_in[3,:] == 1.):
    #         pc_in[3,:] = 1.

    # T_to_prophesee_left = torch.tensor([[ 9.9999e-01, -1.2158e-04, -4.6864e-03, -1.2023e-01],
    #                          [ 1.2021e-04,  1.0000e+00, -2.9335e-04,  8.3630e-04],
    #                          [ 4.6865e-03,  2.9279e-04,  9.9999e-01,  1.0119e-03],
    #                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    # # T_to_prophesee_left = torch.tensor([[ 9.9999e-01, -6.6613e-04, -3.5103e-03, -1.2018e-01],
    # #                          [ 6.6661e-04,  1.0000e+00,  1.3561e-04,  9.1033e-04],
    # #                          [ 3.5102e-03, -1.3795e-04,  9.9999e-01, -4.3059e-04],
    # #                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    # pc_in = torch.matmul(T_to_prophesee_left, pc_in)

    # calib = torch.tensor([1034.86278431, 1033.47800271, 629.70125104, 357.60071019], device='cuda:0', dtype=torch.float)
    # # calib = torch.tensor([1034.39696079, 1034.73607278, 636.27410756, 364.73952748], device='cuda:0', dtype=torch.float)
    # sparse_depth = depth_generation(pc_in.cuda(), (720, 1280), calib, 3., 5, device='cuda:0')
    # from depth_completion import sparse_to_dense
    # from utils import enhanced_depth_line_extract, remove_noise
    # sparse_depth_denoise = remove_noise(sparse_depth.cpu(), radius=5, threshold=5)
    # dense_depth = sparse_to_dense(sparse_depth_denoise[0, ...].cpu().numpy() * 10, 10.)
    # dense_depth_vis = (dense_depth / 10. * 255).astype(np.uint8)
    # sparse_depth = (sparse_depth[0].cpu().numpy()*255).astype(np.uint8)
    # sparse_depth_denoise = (sparse_depth_denoise[0].cpu().numpy()*255).astype(np.uint8)
    # cv2.imwrite(f"./visualization/{idx:05d}_1_sparse_depth.png", sparse_depth)
    # cv2.imwrite(f"./visualization/{idx:05d}_2_sparse_depth_denoise.png", sparse_depth_denoise)
    # cv2.imwrite(f"./visualization/{idx:05d}_3_dense_depth.png", dense_depth_vis)

    # ## denoise
    # from Kinect_Smoothing.kinect_smoothing import HoleFilling_Filter, Denoising_Filter
    # noise_filter = Denoising_Filter(flag='anisotropic', 
    #                                 depth_min=0.01,
    #                                 depth_max=10., 
    #                                 niter=10,
    #                                 kappa=100,
    #                                 gamma=0.25,
    #                                 option=1)
    # image_frame = noise_filter.smooth_image(dense_depth)
    # cv2.imwrite(f"./visualization/{idx:05d}_4_dense_depth_denoise.png", (image_frame/10.*255).astype(np.uint8))


    # blur

    # import time
    # begin = time.time()
    # stretched_image, output_image, dense_edge = enhanced_depth_line_extract(dense_depth / 10. * 255)
    # print(time.time() - begin)
    # cv2.imwrite(f"./visualization/{idx:05d}_5_dense_stretched.png", stretched_image)
    # cv2.imwrite(f"./visualization/{idx:05d}_6_dense_output.png", output_image.cpu().numpy()*255)
    # cv2.imwrite(f"./visualization/{idx:05d}_7_dense_edge.png", dense_edge.cpu().numpy()*255)
    # # cv2.imwrite(f"./visualization/{idx:05d}_6_dense_output.png", output_image*255)
    # # cv2.imwrite(f"./visualization/{idx:05d}_7_dense_edge.png", dense_edge*255)

    

