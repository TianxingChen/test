'''
    keys: point_cloud, state, action, img
'''

import zarr
import pdb
import sys
from read_frame import ReadPerFrame
import pickle
from copy import deepcopy
import numpy as np
import os
current_path = os.getcwd()
sys.path.insert(0, current_path + '/../../../')
import shutil
import time


# save_dir = os.path.join(args.root_dir, 'adroit_'+args.env_name+'_expert.zarr')
num_pcd = 1024
num = 43
print(f'pcd: {num_pcd}, num: {num}')

save_dir = f'/data/chentianxing/ctx/txdiffuser/3D-Diffusion-Policy/data/data.zarr'
load_dir =  '../../exp_data_raw'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

# masks = get_detect_and_sam_mask(ds, obs, type=1)


# img_arrays = []
# depth_arrays = []
point_cloud_arrays = []
state_arrays = []
action_arrays = [] # vel
episode_ends_arrays = []
rgb_arrays = []
info_arrays = [] # language instruction
tcp_arrays = []


folder_num, file_num = 0, 0

read_per_frame = ReadPerFrame(load_dir)

total_count = 0
while os.path.isdir(load_dir+f'/{folder_num}') and folder_num < num:
    file_num = 0

    point_cloud_sub_arrays = []
    state_sub_arrays = []
    action_sub_arrays = [] # vel
    episode_ends_sub_arrays = []
    # info_sub_arrays = [] # language instruction
    
    while os.path.exists(load_dir+f'/{folder_num}'+f'/{file_num}.pickle'):
        print(folder_num, file_num)

        obs = read_per_frame.get_obs_per_frame(folder_num, file_num)   
        # odict_keys(['agent', 'extra', 'camera_param', 'image', 'info', 'frame_id', 'pcd', 'rgb', 'pcd_info'])
        point_cloud = None
        pcd = obs['point_cloud']
        state = obs['state']
        action = obs['action']

        point_cloud_sub_arrays.append(pcd)
        # rgb_sub_arrays.append(sample_pcd_rgb)
        state_sub_arrays.append(state)
        action_sub_arrays.append(action) # vel
        # info_sub_arrays.append(info)

        file_num += 1
        total_count += 1
    
    folder_num += 1

    episode_ends_arrays.append(deepcopy(total_count))
    point_cloud_arrays.extend(point_cloud_sub_arrays)
    state_arrays.extend(state_sub_arrays)
    action_arrays.extend(action_sub_arrays)

state_arrays = np.stack(state_arrays, axis=0)
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

zarr_root = zarr.group(save_dir)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)


state_chunk_size = (100, state_arrays.shape[1])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
action_chunk_size = (100, action_arrays.shape[1])

zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
