
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from os.path import dirname, abspath, join
import os
current_dir = dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir)
sys.path.append(current_dir+'/..')
from owl_vit import OWL_VIT

class Detect_and_Seg:
    def __init__(self, device="cuda:0") -> None:
        module_path = abspath(__file__)
        package_dir = dirname(module_path)

        sam_checkpoint = package_dir + "/checkpoints/sam_vit_h.pth"
        model_type = "vit_h"

        self.vit_model = OWL_VIT(device=device)
        print('owl_vit is loaded successfully')

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        print('sam predictor is loaded successfully')

        self.auto_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        # predictor.set_image(image)
    
    def set_score_threshold(self, score_threshold):
        self.vit_model.set_score_threshold(score_threshold)

    def auto_generate_mask(self, image):
        masks = self.auto_generator.generate(image)
        # ['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
        '''
            * `segmentation` : the mask
            * `area` : the area of the mask in pixels
            * `bbox` : the boundary box of the mask in XYWH format
            * `predicted_iou` : the model's own prediction for the quality of the mask
            * `point_coords` : the sampled input point that generated this mask
            * `stability_score` : an additional measure of mask quality
            * `crop_box` : the crop of the image used to generate this mask in XYWH format
        '''
        return masks

    def get_value_by_auto_generate_mask(self, image, text_queries):
        masks = self.auto_generate_mask(image)
        for i in range(len(masks)):
            box = masks['bbox']
            masks, scores, logits = self.segment_box()


    def detect_and_seg(self, image, text, seg_type='box', type=0, threshold=-1): # 0 over 
        if threshold != -1:
            self.set_score_threshold(threshold)
        
        boxes, labels, scores = self.detect(image, text, type=type)
        masks, scores, logits = self.get_segment(image, boxes, type=seg_type)
        return masks, scores, logits, boxes

    def detect(self, image, text, type=0, transform=True):
        boxes, labels, scores = self.vit_model.get_box(image, text, type=type, transform=transform)
        return boxes, labels, scores
            
    def get_segment(self, image, input, type='box'):
        image_shape = image.shape
        self.predictor.set_image(image)

        if type == 'box':
            boxes = np.array(input)
            masks, scores, logits = self.segment_box(boxes, multimask_output=True)
        else:
            point_coords = []
            for i in range(len(input)):
                cx, cy, w, h = input[i]
                cx += w/2
                cy += h/2
                point_coords.append([cx, cy])
            point_coords = np.array(point_coords)
            masks, scores, logits = self.segment_dot(point_coords, multimask_output=True)
        return masks, scores, logits
        

    def segment_dot(self, point_coords, point_labels=np.array([1]), multimask_output=True):
        masks, scores, logits = [], [], []
        for i in range(point_coords.shape[0]):
            mask, score, logit = self.predictor.predict(
                point_coords=np.array([point_coords[i]]),
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            idx = self.find_max_indices(score)
            mask, score, logit = [mask[idx]], [score[idx]], [logit[idx]]
            masks.append(mask)
            scores.append(score)
            logits.append(logit)
        return masks, scores, logits
    
    def segment_box(self, boxes, multimask_output=True):
        masks, scores, logits = [], [], []
        for i in range(boxes.shape[0]):
            cx, cy, w, h = boxes[i]
            box = [cx, cy, cx+w, cy+h]
            mask, score, logit = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([box[:]]),
                multimask_output=multimask_output,
            )
            idx = self.find_max_indices(score)
            mask, score, logit = [mask[idx]], [score[idx]], [logit[idx]]
            masks.append(mask)
            scores.append(score)
            logits.append(logit)
        return masks, scores, logits 


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        # w, h = box[2] - box[0], box[3] - box[1]
        w, h = box[2], box[3]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
            coords = np.array(coords)
            ax.scatter(coords[:, 0], coords[:, 1], color='red', marker='*', s=125, edgecolor='white', linewidth=1.25)   
        ax.imshow(img)

    def find_max_indices(self, lst):
        max_value = max(lst)
        max_indices = [i for i, value in enumerate(lst) if value == max_value]
        return max_indices[0]

    def merge_mask_or(self, masks):
        n, m = masks[0][0].shape
        res = np.zeros_like(masks[0][0])
        for i in range(len(masks)):
            for j in range(len(masks[i])):
                res |= masks[i][j]
        return res
    
    def merge_mask_and(self, masks):
        n, m = masks[0][0].shape
        res = np.ones_like(masks[0][0])
        for i in range(len(masks)):
            for j in range(len(masks[i])):
                print(i, j)
                res = res & masks[i][j]
        return res

    def mask_inv(self, masks):
        n, m = masks[0][0].shape
        for i in range(len(masks)):
            for j in range(len(masks[i])):
                for x in range(n):
                    for y in range(m):
                        if masks[i][j][x][y] == True:
                            masks[i][j][x][y] = False
                        else:
                            masks[i][j][x][y] = True
        return masks

def load_image(url):
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def sp_gif():
    from PIL import Image

    def split_gif(input_gif, output_folder):
        # 使用Pillow打开GIF文件
        with Image.open(input_gif) as img:
            # 获取GIF中的帧数
            total_frames = img.n_frames

            # 保存每一帧为单独的图片
            for frame_count in range(total_frames):
                print(frame_count, total_frames)
                img.seek(frame_count)
                img.save(f"{output_folder}/frame_{frame_count:03d}.png")
    # 使用函数
    input_gif_path = '/data/chentianxing/ctx/txdiffuser/third_party/PhysiLogic-arti_pick_and_place/exp_data/test/exp_data_fre_30_10/_gif/000_left_camera.gif'  # 替换为你的GIF文件路径
    output_folder_path = './load_dir/left_camera'  # 替换为你想保存帧的文件夹路径
    split_gif(input_gif_path, output_folder_path)



if __name__ == "__main__":
    # sp_gif()

    # config
    sam_checkpoint = "checkpoints/sam_vit_h.pth"
    model_type = "vit_h"
    device = "cuda:0"

    # url = '../images/top_down_camera_orig.jpg'
    url = './load_dir/robot.png'

    ds = Detect_and_Seg(device=device)
    ds.set_score_threshold(0.01)

    # text_queries = ["drawer handle"]
    # text_queries = ['robot hand', 'infrared digital thermonmeter'] # infrared digital thermonmeter
    text_queries = ['white table'] # infrared digital thermonmeter
    
    image = load_image(url)
    masks, scores, logits, boxes = ds.detect_and_seg(image, text_queries, type=1, seg_type='dot') # max
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    res = ds.merge_mask_or(masks)
    ds.show_mask(res, plt.gca())
    plt.title(f"digital thermonmeter", fontsize=18)
    plt.savefig(f'./test_image/robot.png')
    print('robot is saved')