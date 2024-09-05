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
    return torch.tensor([339.69174, 340.96127, 305.8753, 235.33929])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", 
                    default="/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/VECtor",
                    help="Root path to the dataset", 
                    type=str)
    ap.add_argument("--sequence",
                    default="corridors_dolly1",
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


    sequences = sorted(glob.glob(f"{event_paths}/*.npy"))
    for idx in tqdm.tqdm(range(len(sequences))):
        event_path = sequences[idx]
        sequence = event_path.split('/')[-1].split('_')[-1].split('.')[0]
        pc_path = os.path.join(pc_paths, f'point_cloud_{int(sequence)+1:05d}.h5')

        events = np.load(event_path)
        event_time_image = events
        event_time_image[event_time_image<0]=0
        event_time_image = np.concatenate((np.zeros([event_time_image.shape[0], event_time_image.shape[1], 1]), event_time_image), axis=2)
        event_time_image = (event_time_image / np.max(event_time_image) * 255).astype(np.uint8)
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

        calib = get_calib(args.sequence)
        calib = calib.cuda().float()
        sparse_depth = depth_generation(pc_in.cuda(), (480, 640), calib, 3., 5, device='cuda:0')
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

    

