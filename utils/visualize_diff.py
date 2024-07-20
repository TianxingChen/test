import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
# from constants import DT

DT = 0.01

import IPython
e = IPython.embed

jointInOrder = ['knee', 'lumbar_yaw', 'lumbar_pitch','lumbar_roll', 'neck_yaw', 'neck_pitch', 'neck_roll',  # 0 - 6
                'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_yaw', 'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll',  # 7 - 13
                'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_yaw', 'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll',  # 14 - 20
                'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',  # 21 - 25
                'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']  # 26 - 30
STATE_NAMES = jointInOrder

def visualize_diff(pred_list, gt_list, save_path=None):
    label1, label2, label3 = 'actionPred', 'actionGT', 'difference (pred-gt)'

    pred_val = np.array(pred_list) # ts, dim
    gt_val = np.array(gt_list)
    num_ts, num_dim = pred_val.shape

    part_idx = list(range(num_dim))
    num_dim = len(part_idx)

    h, w = 3, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    # all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    all_names = STATE_NAMES

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(pred_val[:, part_idx[dim_idx]], label=label1)
        ax.set_title(f'Joint {part_idx[dim_idx]}: {all_names[part_idx[dim_idx]]}')
        ax.legend()

    # plot arm gt_val
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(gt_val[:, part_idx[dim_idx]], label=label2)
        ax.legend()

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(pred_val[:, part_idx[dim_idx]] - gt_val[:, part_idx[dim_idx]], label=label3)
        ax.legend()

    plt.tight_layout()

    plot_path = save_path
    plt.savefig(plot_path)
    # plt.show()
    print(f'Saved pred_val plot to: {plot_path}')
    plt.close()
    plt.clf()
