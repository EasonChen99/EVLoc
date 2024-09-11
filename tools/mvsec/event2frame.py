import numpy as np
import os
import h5py
import argparse
import tqdm
import cv2
from utils import find_near_index

parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/MVSEC', metavar='PARAMS', help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/MVSEC', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()


save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
  os.makedirs(save_path)

count_dir = os.path.join(save_path, 'event_frames_ours_denoise_pre_100000', 'left')
if not os.path.exists(count_dir):
  os.makedirs(count_dir)
  
gray_dir = os.path.join(save_path, 'gray', 'left')
if not os.path.exists(gray_dir):
  os.makedirs(gray_dir)
  

class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)], shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0, poses_ts=0, dt_time_temp=0):
        print(input_event.shape, poses_ts.shape)

        # print(timestamps[0], timestamps[-1])
        # print(timestamps[-1]-timestamps[0])
        # print(poses_ts[0])
        # print(poses_ts[1] - poses_ts[0])
        # sys.exit()

        split_interval = poses_ts.shape[0]

        td_img_c = np.zeros((self.height, self.width, 2), dtype=np.float32)

        t_index = 0
        for i in tqdm.tqdm(range(split_interval)):
            idx_start, idx_cur, idx_end = find_near_index(poses_ts[i], input_event[:, 2], time_window=dt_time_temp*2)
            # print(idx_cur-idx_start)
            # print(poses_ts[i])
            # print(idx_cur, timestamps[idx_cur])

            if idx_cur - idx_start > 0:
                td_img_c.fill(0)

            r = 6
            for id in range(idx_cur-idx_start):
                idx = int(id + idx_start)
                y, x = int(input_event[idx, 1]), int(input_event[idx, 0])
                if input_event[idx, 3] > 0:
                    patch = td_img_c[y-r:y+r+1, x-r:x+r+1, 0]
                    patch = np.where(patch>=td_img_c[y,x,0], patch-(input_event[idx, 2]-patch)/15., patch)
                    td_img_c[y-r:y+r+1, x-r:x+r+1, 0] = patch
                    td_img_c[y, x, 0] = input_event[idx, 2]
                else:
                    patch = td_img_c[y-r:y+r+1, x-r:x+r+1, 1]
                    patch = np.where(patch>=td_img_c[y,x,1], patch-(input_event[idx, 2]-patch)/15., patch)
                    td_img_c[y-r:y+r+1, x-r:x+r+1, 1] = patch
                    td_img_c[y, x, 1] = input_event[idx, 2]
            td_img_c[td_img_c < 0] = 0

            t_index = t_index + 1

            np.save(os.path.join(count_dir, f'event_frame_{i:05d}'), td_img_c)


# d_set = h5py.File(args.data_path, 'r')
d_set = h5py.File(os.path.join(args.data_path, args.save_env+"_data.hdf5"), 'r')
raw_data = d_set['davis']['left']['events']
d_set = None

d_set = h5py.File(os.path.join(args.data_path, args.save_env+"_gt.hdf5"), 'r')
poses_ts = d_set['davis']['left']['odometry_ts']
d_set = None

dt_time = 0.05

td = Events(raw_data.shape[0])
# Events
td.generate_fimage(input_event=raw_data, poses_ts=poses_ts, dt_time_temp=dt_time)
raw_data = None


print('Encoding complete!')
