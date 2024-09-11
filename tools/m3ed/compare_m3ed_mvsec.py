import os
import sys
import h5py


mvsec_data_path = f"/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/MVSEC/outdoor_day1_data.hdf5"
m3ed_data_path = f"/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/M3ED/original/Falcon/flight_3/falcon_indoor_flight_3_data.h5"

mvsec_d_set = h5py.File(mvsec_data_path, 'r')
m3ed_d_set = h5py.File(m3ed_data_path, 'r')
print(mvsec_d_set.keys())
print(m3ed_d_set.keys())

mvsec_event_left = mvsec_d_set['davis/left']
print(mvsec_event_left.keys())
m3ed_event_left = m3ed_d_set['prophesee/left']
print(m3ed_event_left.keys())
m3ed_calib = m3ed_event_left['calib']
print(m3ed_calib.keys())
print(m3ed_calib['intrinsics'][:])


