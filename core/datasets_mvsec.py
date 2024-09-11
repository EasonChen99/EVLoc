import os
import csv
import h5py
from math import radians
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import mathutils
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from core.utils_point import quaternion_from_matrix, invert_pose, rotate_forward, rotate_back

def get_calib_m3ed(sequence):
    if sequence in ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'indoor_flying4']:
        return torch.tensor([199.65301231, 199.65301231, 177.43276376, 126.81215684])
    else:
        raise TypeError("Sequence Not Available")

def get_left_right_T(sequence):
    if sequence in ['indoor_flying1', 'indoor_flying2', 'indoor_flying3', 'indoor_flying4']:
        return torch.tensor([[0.9999285439274112, 0.011088072985503046, -0.004467849222081981, -0.09988137641750752],
                             [-0.011042817783611191, 0.9998887260774646, 0.01002953830336461, -0.0003927067773089277],
                             [0.004578560319692358, -0.009979483987103495, 0.9999397215256256, 1.8880107752680777e-06],
                             [0.0, 0.0, 0.0, 1.0]])
    else:
        raise TypeError("Sequence Not Available")

class DatasetMVSEC(Dataset):
    def __init__(self, dataset_dir, suffix, method, ran, max_t=0.5, max_r=5., split='test', device='cuda:0', test_sequence='indoor_flying4'):
        super(DatasetMVSEC, self).__init__()
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.suffix = suffix
        self.method = method
        self.ran = ran
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}

        self.all_files = []
        
        scene_list = [
                      'indoor_flying1', 
                      'indoor_flying2', 
                    #   'indoor_flying3', 
                    #   'indoor_flying4'
                     ]
        
        for dir in scene_list:
            self.GTs_R[dir] = []
            self.GTs_T[dir] = []

            pose_path = os.path.join(self.root_dir, dir, dir+"_gt.hdf5")
            poses = h5py.File(pose_path,'r')
            Ln_T_L0 = poses['davis']['left']['odometry'] 

            for idx in range(Ln_T_L0.shape[0]):
                # if idx + 70 < Ln_T_L0.shape[0] and idx > 70: #M3ED
                # if idx + 140 < Ln_T_L0.shape[0] and idx > 140: #MVSEC_Indoor_Flying
                if idx + 150 < Ln_T_L0.shape[0] and idx > 150:
                # if idx + 250 < Ln_T_L0.shape[0] and idx > 250:
                    if not os.path.exists(os.path.join(self.root_dir, dir, "local_maps", f"point_cloud_{idx:05d}"+'.h5')):
                        continue
                    if not os.path.exists(os.path.join(self.root_dir, dir, f"event_frames_{self.method}_{self.ran}_{self.suffix}", 'left', f"event_frame_{idx:05d}"+'.npy')):
                        continue
                    if dir == test_sequence and split.startswith('test'):
                        self.all_files.append(os.path.join(dir, f"event_frames_{self.method}_{self.ran}_{self.suffix}", 'left', f"{idx:05d}"))
                    elif (not dir == test_sequence) and split == 'train':
                        self.all_files.append(os.path.join(dir, f"event_frames_{self.method}_{self.ran}_{self.suffix}", 'left', f"{idx:05d}"))
                        # self.all_files.append(os.path.join(dir, f"event_frames_{self.method}_{self.ran}_{self.suffix}", 'right', f"{idx:05d}"))
                    
                    R = quaternion_from_matrix(torch.tensor(Ln_T_L0[idx]))
                    T = Ln_T_L0[idx][:3, 3]
                    GT_R = np.asarray(R)
                    GT_T = np.asarray(T)
                    self.GTs_R[dir].append(GT_R)
                    self.GTs_T[dir].append(GT_T)
                else:
                    continue
        
        self.test_RT = []
        if split == 'test':
            test_RT_file = os.path.join(self.root_dir, f'test_RT_seq_{test_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-min(max_t, 0.1), min(max_t, 0.1))
                    test_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])
                    self.test_RT.append([i, transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])

            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"
    
    
    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def __len__(self):
        return len(self.all_files)    

    def __getitem__(self, idx):
        item = self.all_files[idx]
        run = str(item.split('/')[0])
        camera = str(item.split('/')[2])
        timestamp = str(item.split('/')[3])

        event_frame_path = os.path.join(self.root_dir, run, f"event_frames_{self.method}_{self.ran}_{self.suffix}", camera, 'event_frame_'+timestamp+'.npy')
        # pc_path = os.path.join(self.root_dir, run, "local_maps", "point_cloud_"+f"{int(timestamp)+1:05d}"+'.h5')
        pc_path = os.path.join(self.root_dir, run, "local_maps", "point_cloud_"+f"{int(timestamp):05d}"+'.h5')

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
        else:
            raise TypeError("Wrong PointCloud shape")

        if camera == 'right':
            T_to_prophesee_left = get_left_right_T(run)
            pc_in = torch.matmul(T_to_prophesee_left, pc_in)

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[0, :] *= -1
        
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)

        if self.split == 'train':
            R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        event_frame = np.load(event_frame_path)
        event_time_frame = torch.tensor(event_frame).float()
        event_time_frame[event_time_frame<0] = 0
        event_time_frame /= torch.max(event_time_frame)
        event_time_frame = event_time_frame.unsqueeze(-1)

        if event_time_frame.shape[2] <= 4:
            event_time_frame = F.to_pil_image(event_time_frame.permute(2, 0, 1))
            if h_mirror:
                event_time_frame = event_time_frame.transpose(Image.FLIP_LEFT_RIGHT)
            event_time_frame = event_time_frame.rotate(img_rotation)
            # event_time_frame = F.rotate(event_time_frame, img_rotation)
            event_time_frame = F.to_tensor(event_time_frame)
            event_frame = event_time_frame
        else:
            event_frames = []
            for i in range(0, event_time_frame.shape[2], 4):
                event_time_frame_sub = F.to_pil_image(event_time_frame[:, :, i:i+4].permute(2, 0, 1))
                if h_mirror:
                    event_time_frame_sub = event_time_frame_sub.transpose(Image.FLIP_LEFT_RIGHT)
                event_time_frame_sub = event_time_frame_sub.rotate(img_rotation)
                event_time_frame_sub = F.to_tensor(event_time_frame_sub)
                event_frames.append(event_time_frame_sub)
            event_frame = torch.cat(event_frames, dim=0)

        if self.split != 'test':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-min(self.max_t, 0.1), min(self.max_t, 0.1))
        else:
            initial_RT = self.test_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        calib = get_calib_m3ed(run)
        if h_mirror:
            calib[2] = event_frame.shape[2] - calib[2]


        sample = {'event_frame': event_frame, 'point_cloud': pc_in, 
                  'calib': calib, 'tr_error': T, 'rot_error': R}

        return sample        