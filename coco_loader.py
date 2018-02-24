import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
from PIL import Image
import cv2
import Mytransforms

def read_data_file(file_dir):

    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists

def read_json_file(file_dir):
    """
        filename: JSON file

        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []

    for info in data:
        kpt = []
        center = []
        scale = []
        lists = info['info']
        for x in lists:
           kpt.append(x['keypoints'])
           center.append(x['pos'])
           scale.append(x['scale'])
        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
    fp.close()

    return kpts, centers, scales

def generate_heatmap(heatmap, kpt, stride, sigma):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] > 1: # not labeled
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j] += math.exp(-dis)
                    if heatmap[h][w][j] > 1:
                        heatmap[h][w][j] = 1

    return heatmap

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i] - 1
            b = vec_pair[1][i] - 1
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector


def transform_joints(kpts):
    '''
    OURS
    param.model(id).part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                             'Lsho', 'Lelb', 'Lwri', ...
                             'Rhip', 'Rkne', 'Rank', ...
                             'Lhip', 'Lkne', 'Lank', ...
                             'Reye', 'Leye', 'Rear', 'Lear', 'pt19'};


    '''

    COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]

    if len(kpts) == 0:
        return kpts

    new_kpts = np.zeros(
        (len(kpts), len(COCO_to_ours_1), 3),
        dtype=np.float32
    )
    num = len(new_kpts)
    length = len(new_kpts[0])
    for i in range(num):
        for j in range(length):
            new_kpts[i][j][0] = (kpts[i][COCO_to_ours_1[j] - 1][0] + kpts[i][COCO_to_ours_2[j] - 1][0]) * 0.5
            new_kpts[i][j][1] = (kpts[i][COCO_to_ours_1[j] - 1][1] + kpts[i][COCO_to_ours_2[j] - 1][1]) * 0.5

            if kpts[i][COCO_to_ours_1[j] - 1][2] == 2 or kpts[i][COCO_to_ours_2[j] - 1][2] == 2:
                new_kpts[i][j][2] = 2
            else:
                new_kpts[i][j][2] = kpts[i][COCO_to_ours_1[j] - 1][2] and kpts[i][COCO_to_ours_2[j] - 1][2]

    return new_kpts


class coco_loader(data.Dataset):

    def __init__(self, file_dir, stride, transformer=None):

        self.img_list = read_data_file(file_dir[0])
        self.mask_list = read_data_file(file_dir[1])
        self.kpt_list, self.center_list, self.scale_list = read_json_file(file_dir[2])
        self.stride = stride
        self.transformer = transformer

        self.vec_pair = [[2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16],
                         [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]
        self.theta = 1.0
        self.sigma = 7.0
        '''
          1 'nose',
          2' neck',
          3 'Rsho', 
          4 'Relb', 
          5 'Rwri', ...
          6 'Lsho', 
          7 'Lelb', 
          8 'Lwri', ...
          9 'Rhip', 
          10'Rkne', 
          11'Rank', ...
          12'Lhip',
          13'Lkne', 
          14'Lank', ...
          15'Reye', 
          16'Leye', 
          17'Rear', 
          18'Lear', 'pt19'}; 
          
        self.vec_pair = [[2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16],
                         [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]
        '''

    def __getitem__(self, index):

        img_path = '/home/xiangyu/data/coco/images/train2014/'+ self.img_list[index]

        img = np.array(cv2.imread(img_path), dtype=np.float32)
        mask_path = self.mask_list[index]
        mask = np.load(mask_path)
        mask = np.array(mask, dtype=np.float32)

        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        #kpt = transform_joints(kpt)

        img, mask, kpt, center = self.transformer(img, mask, kpt, center, scale)

        height, width, _ = img.shape

        mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape((height / self.stride, width / self.stride, 1))

        heatmap = np.zeros((height / self.stride, width / self.stride, len(kpt[0]) + 1), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, kpt, self.stride, self.sigma)
        heatmap[:,:,-1] = 1.0 - np.max(heatmap[:,:,:-1], axis=2) # for background
        heatmap = heatmap * mask

        vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
        cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0])), dtype=np.int32)

        vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
        vecmap = vecmap * mask

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) # mean, std
        mask = Mytransforms.to_tensor(mask)
        heatmap = Mytransforms.to_tensor(heatmap)
        vecmap = Mytransforms.to_tensor(vecmap)

        # kpts to tensor
        #kpt = np.array(kpt)
        #kpt = torch.from_numpy(kpt)

        return img, heatmap, vecmap, mask, kpt

    def __len__(self):

        return len(self.img_list)
