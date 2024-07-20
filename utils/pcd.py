import numpy as np
import random
import pytorch3d.ops as torch3d_ops
import torch
import time
import pdb

def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc

def tanslation_point_cloud(depth_map, rgb_image, camera_intrinsic, cam2world_matrix, view=True, mask=None):
    depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
    rows, cols = depth_map.shape[0], depth_map.shape[1]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    z = depth_map
    x = (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
    y = (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
    points = np.dstack((x, y, z))
    per_point_xyz = points.reshape(-1, 3)
    line_masks = mask.reshape(-1)
    per_point_rgb = rgb_image.reshape(-1, 3)
    # view_point_cloud_parts(per_point_xyz, actor_seg)
    point_xyz = []
    point_rgb = []
    
    point_xyz = per_point_xyz[np.where(line_masks)]
    point_rgb = per_point_rgb[np.where(line_masks)]
    # point_xyz = per_point_xyz
    # point_rgb = per_point_rgb
    pcd_camera = np.array(point_xyz)
    point_rgb = np.array(point_rgb)
    pcd_world = pc_camera_to_world(pcd_camera, cam2world_matrix)
    return pcd_world, point_rgb


def get_point_cloud(obs, masks=None):
    camera_params = obs["camera_param"]
    images = obs["image"]
    camera_dicts = camera_params
    res = dict()
    res_rgb = dict()
    for camera_name in camera_dicts:
        # print(camera_name)
        if camera_name == 'hand_camera':
            continue
        camera_intrinsic = camera_dicts[camera_name]["intrinsic_cv"]
        cam2world_matrix = camera_dicts[camera_name]["cam2world_gl"]
        Rtilt_rot = cam2world_matrix[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        Rtilt_trl = cam2world_matrix[:3, 3]
        cam2_wolrd = np.eye(4)
        cam2_wolrd[:3, :3] = Rtilt_rot
        cam2_wolrd[:3, 3] = Rtilt_trl
        camera_dicts[camera_name]["cam2world"] = cam2_wolrd
        camera_image = images[camera_name]
        camera_rgb = camera_image["rgb"]
        camera_depth = camera_image["depth"]
        mask = masks[camera_name]
        # print("camera_name: ", camera_name)
        point_cloud_world, per_point_rgb = tanslation_point_cloud(camera_depth, camera_rgb,
                                                                    camera_intrinsic, cam2_wolrd,
                                                                    mask=mask)
        res[camera_name] = point_cloud_world
        res_rgb[camera_name] = per_point_rgb # !!!!!!!!!!!!!!!!
        # camera_dicts[camera_name]["per_seg_mask"] = per_seg_mask
        # camera_dicts[camera_name]["rgb"] = camera_rgb
        # camera_dicts[camera_name]["depth"] = camera_depth
        # camera_dicts[camera_name]["segmentation"] = actor_seg
        # camera_dicts[camera_name]["camera_intrinsic"] = camera_intrinsic
        # view_point_cloud_parts(point_cloud=point_cloud_world, mask=seg_mask)
        # view_point_cloud_parts(point_cloud=point_cloud_world, rgb=per_point_rgb)
    return res, res_rgb

def random_sample_pcd(pcd, pcd_rgb, pcd_info=None, num=10000):

    # 1: table
    # 2. object on the table
    # 3: agent
    # 4: cabinet
    # 5: handle
    # 6. apple


    # import pdb
    # pdb.set_trace()    
    t = 0.1 # crop ground
    bool_mask = pcd[:, 2] > t
    idx = np.where(bool_mask)
    tmp = pcd[idx]
    tmp_rgb = pcd_rgb[idx]

    if pcd_info is not None:
        tmp_info = pcd_info[idx]


    # bool_mask = (tmp_info != 4) & (tmp_info != 5)
    bool_mask = (tmp_info == 3) | (tmp_info == 6)
    idx = np.where(bool_mask)
    tmp = pcd[idx]
    tmp_rgb = pcd_rgb[idx]

    if pcd_info is not None:
        tmp_info = pcd_info[idx]


    len = tmp.shape[0]
    idx = random.sample(range(len), num)
    if pcd_info is None:
        return tmp[idx], tmp_rgb[idx]
    else:
        return tmp[idx], tmp_rgb[idx], tmp_info[idx]


def farthest_point_sampling(point_cloud, num=10000, device='cuda:0'):
    pcd = point_cloud[:, :3]
    array = np.array([pcd])

    # 然后，选择其中的前三个维度
    points_array = array[..., :3]

    # 最后，将这个 ndarray 对象转换为 PyTorch 张量，并发送到指定的设备
    p = torch.tensor(points_array).to(device)

    _, idx = torch3d_ops.sample_farthest_points(points=p, K=num)
    idx = idx[0].to('cpu')
    
    return point_cloud[idx]


def random_with_fps(pcd, pcd_rgb, pcd_info=None, num=10000, device='cuda:0'):
    len1 = max(pcd.shape[0] // 10, num)
    if pcd_info is None:
        tmp_pcd, tmp_pcd_rgb = random_sample_pcd(pcd, pcd_rgb, pcd_info, len1)  
    else:
        tmp_pcd, tmp_pcd_rgb, tmp_info = random_sample_pcd(pcd, pcd_rgb, pcd_info, len1)  
    return farthest_point_sampling(tmp_pcd, tmp_pcd_rgb, tmp_info, num=num, device=device)
