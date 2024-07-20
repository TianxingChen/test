import pickle
import sys
import os
import shutil
import pdb
import imageio

class ReadPerFrame:
    def __init__(self, save_path) -> None:
        self.save_path = save_path
        self.folder_path = ''
        self.scene_num = 0
        self.current_frame = 0
    
    def read_obs_per_frame(self, scene_num, current_frame): # ctx
        folder_path = self.save_path + f'/{scene_num}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = f'{current_frame}'
        file_path = folder_path + '/' + file_name + '.pickle'

        print(f'file path: {file_path}')

        # 从pickle文件中读取字典数据
        with open(file_path, 'rb') as file:
            obs = pickle.load(file)

        size_bytes = sys.getsizeof(obs)
        size_mb = size_bytes / (1024 * 1024)
        print(f'Successfully saved to: {file_path}, size: {size_mb} MB')
        return obs
    
    def get_obs_per_frame(self, scene_num, current_frame, info=False): # ctx
        folder_path = self.save_path + f'/{scene_num}'
        file_name = f'{current_frame}'
        # file_name = '_info'
        file_path = folder_path + '/' + file_name + '.pickle'

        if info:
            print(f'file path: {file_path}')

        # 从pickle文件中读取字典数据
        with open(file_path, 'rb') as file:
            obs = pickle.load(file)
        return obs

    def save_gif(self, scene_num, st_frame, ed_frame, camera): # ctx
        images = []
            
        duration = 0.002  # 每帧之间的延迟时间（以秒为单位）
        size_bytes = 0

        output_path = self.save_path + f'/{scene_num}/_{camera}_{duration}.gif'

        for current_frame in range(st_frame, ed_frame, 5): # render per 5 frame
             # 获取列表对象的字节大小
            size_gb = size_bytes / (1024**3)  # 将字节大小转换为GB
            size_mb = size_bytes / (1024**2)  # 将字节大小转换为GB
            print(f'loading {current_frame} / {ed_frame},  images size: {size_mb:.2f} MB', end='\r')
            folder_path = self.save_path + f'/{scene_num}'
            file_name = f'{current_frame}'
            file_path = folder_path + '/' + file_name + '.pickle'

            # 从pickle文件中读取字典数据
            with open(file_path, 'rb') as file:
                obs = pickle.load(file)
        
            image = obs['image'][camera]['rgb']
            size_bytes += image.nbytes
            images.append(image)
        print('\nsaving fig')
        imageio.mimsave(output_path, images, duration=duration)
        print(f'saved to {output_path}')
    

if __name__ == "__main__":
    save_path = '/data/chentianxing/ctx/txdiffuser/third_party/PhysiLogic-arti_pick_and_place/exp_data/test/exp_data_fre_40'
    arr = []
    a = ReadPerFrame(save_path)

    scene_num, current_num = 0, 0
    res = a.get_obs_per_frame(scene_num, current_num)
    arr.append(res['agent']['qpos'])
    
    scene_num, current_num = 0, 1
    res = a.get_obs_per_frame(scene_num, current_num)
    arr.append(res['agent']['qpos'])

    scene_num, current_num = 0, 2
    res = a.get_obs_per_frame(scene_num, current_num)
    arr.append(res['agent']['qpos'])

    scene_num, get_obs_per_frame = 0, 3
    res = a.get_obs_per_frame(scene_num, current_num)
    arr.append(res['agent']['qpos'])

    for i in range(len(arr)):
        print(arr[i])
    pdb.set_trace()