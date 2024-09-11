import os
import glob
import h5py
import argparse
import numpy as np
import cv2
import torch
from utils import depth_generation
from utils_point import overlay_imgs
import tqdm

def get_calib(sequence):
    return torch.tensor([199.65301231, 199.65301231, 177.43276376, 126.81215684])

def get_left_right_T(sequence):
    return torch.tensor([[0.9999285439274112, 0.011088072985503046, -0.004467849222081981, -0.09988137641750752],
                        [-0.011042817783611191, 0.9998887260774646, 0.01002953830336461, -0.0003927067773089277],
                        [0.004578560319692358, -0.009979483987103495, 0.9999397215256256, 1.8880107752680777e-06],
                        [0.0, 0.0, 0.0, 1.0]])

def get_rectification_matrix(sequence):
    projection_matrix = np.array([[199.6530123165822, 0.0, 177.43276376280926],
                                  [0.0, 199.6530123165822, 126.81215684365904],
                                  [0.0, 0.0, 1.0]])
    rectification_matrix = np.array([[0.999877311526236, 0.015019439766575743, -0.004447282784398257],
                                    [-0.014996983873604017, 0.9998748347535599, 0.005040367172759556],
                                    [0.004522429630305261, -0.004973052949604937, 0.9999774079320989]])
    # return np.matmul(projection_matrix, rectification_matrix)
    return projection_matrix

def get_distortion_coeffs(sequence):
    return np.array([-0.048031442223833355, 0.011330957517194437, -0.055378166304281135, 0.021500973881459395])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/MVSEC",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="indoor_flying1",
                    help="Sequence name for processing",
                    type=str)
    ap.add_argument("--method",
                    default="gray____",
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


    sequences = sorted(glob.glob(f"{event_paths}/*.npy"))
    for idx in tqdm.tqdm(range(len(sequences))):
        event_path = sequences[idx]
        sequence = event_path.split('/')[-1].split('_')[-1].split('.')[0]
        pc_path = os.path.join(pc_paths, f'point_cloud_{int(sequence)+1:05d}.h5')

        events = np.load(event_path)
        event_time_image = events
        event_time_image[event_time_image<0]=0
        # event_time_image = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
        event_time_image = (event_time_image / np.max(event_time_image) * 255).astype(np.uint8)
        ## undistort
        h, w = event_time_image.shape[:2]
        rectification_matrix = get_rectification_matrix(args.sequence)
        distortion_coeffs = get_distortion_coeffs(args.sequence)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(rectification_matrix, distortion_coeffs, (w, h), 1, (w, h))
        event_time_image = cv2.undistort(event_time_image, rectification_matrix, distortion_coeffs, None, new_camera_matrix)
        cv2.imwrite(f"./visualization/{idx:05d}_event.png", event_time_image)
        
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

        calib = get_calib(args.sequence)
        calib = calib.cuda().float()
        sparse_depth = depth_generation(pc_in.cuda(), (260, 346), calib, 3., 5, device='cuda:0')
        sparse_depth = overlay_imgs(sparse_depth.repeat(3, 1, 1)*0, sparse_depth[0, ...])
        cv2.imwrite(f"./visualization/{idx:05d}_depth.png", sparse_depth)

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

    

