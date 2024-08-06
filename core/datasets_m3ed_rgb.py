import os
import csv
import h5py
from math import radians
import numpy as np
from PIL import Image
import pandas as pd
import mathutils
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from core.utils_point import quaternion_from_matrix, invert_pose, rotate_forward

def get_calib_kitti(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([1268.56022264, 1267.34933732, 649.36902423, 359.93799316])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([1269.24837363, 1269.16020362, 650.94892772, 368.38236713])
    # elif sequence in ['spot_indoor_building_loop', 'spot_indoor_obstacles', 'spot_indoor_stairs']:
    #     return torch.tensor([1032.0231, 1031.9229,  635.7985,  363.7983])
    else:
        raise TypeError("Sequence Not Available")

def get_prophesee_left_T_rgb(sequence):
    if sequence == "falcon_indoor_flight_1":
        return torch.tensor([[ 0.99995986,  0.00772153,  0.00454551,  0.03025287],
                             [-0.00770454,  0.99996331, -0.0037437,  -0.07599698],
                             [-0.00457425,  0.00370853,  0.99998266, -0.0085445],
                             [ 0.,          0.,         0.,          1.,        ]])
    elif sequence in ["falcon_indoor_flight_2", "falcon_indoor_flight_3"]:
        return torch.tensor([[ 0.99995984,  0.00833675, -0.00328862,  0.03094608],
                             [-0.00835187,  0.99995449, -0.00461065, -0.07603972],
                             [ 0.00325003,  0.00463793,  0.99998396, -0.00790814],
                             [ 0.,          0.,          0.,          1.,        ]])

class DatasetM3ED(Dataset):
    def __init__(self, dataset_dir, suffix, max_t=2., max_r=10., split='test', device='cuda:0', test_sequence='falcon_indoor_flight_3'):
        super(DatasetM3ED, self).__init__()
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.root_dir = dataset_dir
        self.suffix = suffix
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}

        self.all_files = []
        
        scene_list = ['falcon_indoor_flight_1', 
                      'falcon_indoor_flight_2', 
                      'falcon_indoor_flight_3', 
                    #   'spot_indoor_building_loop', 
                    #   'spot_indoor_obstacles',
                     ]
        
        for dir in scene_list:
            self.GTs_R[dir] = []
            self.GTs_T[dir] = []

            pose_path = os.path.join(self.root_dir, dir, "poses.h5")
            poses = h5py.File(pose_path,'r')
            Ln_T_L0 = poses['matrix'][:]

            for idx in range(Ln_T_L0.shape[0]):
                if idx + 70 < Ln_T_L0.shape[0] and idx > 70:
                    if not os.path.exists(os.path.join(self.root_dir, dir, "local_maps", f"point_cloud_{idx:05d}"+'.h5')):
                        continue
                    if not os.path.exists(os.path.join(self.root_dir, dir, "rgb", f"rgb_{idx:05d}"+'.png')):
                        continue
                    if dir == test_sequence and split.startswith('test'):
                        self.all_files.append(os.path.join(dir, "rgb", f"{idx:05d}"))
                    elif (not dir == test_sequence) and split == 'train':
                        self.all_files.append(os.path.join(dir, "rgb", f"{idx:05d}"))
                    
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

            self.test_RT = self.test_RT[:-1]
            assert len(self.test_RT) == len(self.all_files), "Something wrong with test RTs"
    
    
    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def __len__(self):
        return len(self.all_files)    
    
    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = F.hflip(rgb)
            rgb = F.rotate(rgb, img_rotation)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)

        return rgb

    def __getitem__(self, idx):
        item = self.all_files[idx]
        run = str(item.split('/')[0])
        timestamp = str(item.split('/')[2])

        rgb_path = os.path.join(self.root_dir, run, "rgb", 'rgb_'+timestamp+'.png')
        pc_path = os.path.join(self.root_dir, run, "local_maps", "point_cloud_"+timestamp+'.h5')

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
        
        prophesee_left_T_rgb = get_prophesee_left_T_rgb(run)
        pc_in = torch.matmul(prophesee_left_T_rgb.inverse(), pc_in)

        h_mirror = False
        if np.random.rand() > 0.5 and self.split == 'train':
            h_mirror = True
            pc_in[0, :] *= -1
        
        img_rotation = 0.
        if self.split == 'train':
            img_rotation = np.random.uniform(-5, 5)
        
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        rgb = Image.open(rgb_path)
        try:
            rgb = self.custom_transform(rgb, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        if self.split != 'test':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
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

        calib = get_calib_kitti(run)
        if h_mirror:
            calib[2] = rgb.shape[2] - calib[2]


        sample = {'rgb': rgb, 'point_cloud': pc_in, 
                  'calib': calib, 'tr_error': T, 'rot_error': R}

        return sample        