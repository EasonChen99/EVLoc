import numpy as np
import os
import sys
import h5py
import hdf5plugin
import argparse
import tqdm
import cv2
from utils import find_near_index

parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='/home/eason/WorkSpace/EventbasedVisualLocalization/preprocessed_dataset/VECtor', metavar='PARAMS', help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='corridors_dolly1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/VECtor', metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()


save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
  os.makedirs(save_path)

count_dir = os.path.join(save_path, 'event_frames_ours_denoise_pre_100000', 'left')
if not os.path.exists(count_dir):
  os.makedirs(count_dir)
  

class Events(object):
    def __init__(self, width=346, height=260):
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0, t_offset=0, poses=0, dt_time_temp=0):
        td_img_c = np.zeros((self.height, self.width, 2), dtype=np.float32)

        event_ts = input_event['t'][:] + t_offset

        t_index = 0
        for i in tqdm.tqdm(range(len(poses))):
            idx_start, idx_cur, idx_end = find_near_index(poses[i][0]*1e6, event_ts, time_window=dt_time_temp*2)
            # print(idx_cur-idx_start)
            # print(poses_ts[i])
            # print(idx_cur, timestamps[idx_cur])
            # print(event_ts[idx_start], event_ts[idx_cur], event_ts[idx_end])

            if idx_cur - idx_start > 0:
                td_img_c.fill(0)

                r = 6
                for id in range(idx_cur-idx_start):
                    idx = int(id + idx_start)
                    y, x = int(input_event['y'][idx]), int(input_event['x'][idx])
                    if input_event['p'][idx] > 0:
                        patch = td_img_c[y-r:y+r+1, x-r:x+r+1, 0]
                        patch = np.where(patch>=td_img_c[y,x,0], patch-(input_event['t'][idx]-patch)/15., patch)
                        td_img_c[y-r:y+r+1, x-r:x+r+1, 0] = patch
                        td_img_c[y, x, 0] = input_event['t'][idx]
                    else:
                        patch = td_img_c[y-r:y+r+1, x-r:x+r+1, 1]
                        patch = np.where(patch>=td_img_c[y,x,1], patch-(input_event['t'][idx]-patch)/15., patch)
                        td_img_c[y-r:y+r+1, x-r:x+r+1, 1] = patch
                        td_img_c[y, x, 1] = input_event['t'][idx]
                td_img_c[td_img_c < 0] = 0

                t_index = t_index + 1

                # print((td_img_c>0).sum())
                # print(np.max(td_img_c), np.min(td_img_c))
                np.save(os.path.join(count_dir, f'event_frame_{i:05d}'), td_img_c)


d_set = h5py.File(os.path.join(args.data_path, args.save_env, args.save_env+".synced.left_event.hdf5"), 'r')
raw_data = d_set['events']
# ms_to_idx = d_set['ms_to_idx']
t_offset = d_set['t_offset'][:]
d_set = None


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


dt_time = 100000
td = Events(width=640, height=480)
# Events
td.generate_fimage(input_event=raw_data, t_offset= t_offset, poses=poses, dt_time_temp=dt_time)
raw_data = None

print('Encoding complete!')
