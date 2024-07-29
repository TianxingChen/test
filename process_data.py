import pdb, pickle, os
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil

visualize_pcd = False
load_dir = '../720_to_ctx'
folder_num, num = 0, 30
total_count = 0

save_dir = './3D-Diffusion-Policy/data/ctx.zarr'

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

zarr_root = zarr.group(save_dir)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

point_cloud_arrays, episode_ends_arrays, action_arrays, state_arrays = [], [], [], []
while os.path.isdir(load_dir+f'/episode{folder_num}') and folder_num < num:
    file_num = 0

    point_cloud_sub_arrays = []
    state_sub_arrays = []
    action_sub_arrays = [] 
    episode_ends_sub_arrays = []
    # info_sub_arrays = [] # language instruction
    
    while os.path.exists(load_dir+f'/episode{folder_num}'+f'/{file_num}.pkl'):
        print(f'{file_num}, {folder_num}', end='\r')
        with open(load_dir+f'/episode{folder_num}'+f'/{file_num}.pkl', 'rb') as file:
            data = pickle.load(file)
        pcd = data['pcd']['points']
        x, y, z, yaw, pitch, roll, gripper = data['endpose']['x'], data['endpose']['y'], \
                                            data['endpose']['z'], data['endpose']['yaw'], data['endpose']['pitch'], data['endpose']['roll'], data['endpose']['gripper']
        action = np.array([x, y, z, yaw, pitch, roll, gripper])

        point_cloud_sub_arrays.append(pcd)
        state_sub_arrays.append(action)
        action_sub_arrays.append(action)

        if visualize_pcd:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data['pcd']['points'])
            pcd.colors = o3d.utility.Vector3dVector(data['pcd']['colors'])
            o3d.visualization.draw_geometries([pcd])
        file_num += 1
        total_count += 1
        
    folder_num += 1

    episode_ends_arrays.append(deepcopy(total_count))
    point_cloud_arrays.extend(point_cloud_sub_arrays)
    action_arrays.extend(action_sub_arrays)
    state_arrays.extend(state_sub_arrays)

episode_ends_arrays = np.array(episode_ends_arrays)
action_arrays = np.array(action_arrays)
state_arrays = np.array(state_arrays)
point_cloud_arrays = np.array(point_cloud_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
action_chunk_size = (100, action_arrays.shape[1])
state_chunk_size = (100, state_arrays.shape[1])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])


zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
