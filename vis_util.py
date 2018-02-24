import math
import torch
import shutil
import time
import os
import random
import numpy as np
import cv2
import CocoFolder
import Mytransforms
from scipy.ndimage.filters import gaussian_filter
import utils


stride = 8
thre_point = 0.15
thre_line = 0.05
stickwidth = 4
boxsize = 368
padValue = 0.

scale_search = [0.5, 1.0, 1.5, 2.0]

limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

boxsize = 368
scale_search = [0.5, 1.0, 1.5, 2.0]
stride = 8
padValue = 0.
thre1 = 0.1
thre2 = 0.05
stickwidth = 4

# limbSeq = [[3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 14], [1, 2], [2, 9], [2, 12], [2, 3],
#            [2, 6], \
#            [3, 17], [6, 18], [1, 16], [1, 15], [16, 18], [15, 17]]
#
# mapIdx = [[19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37, 38],
#           [39, 40], \
#           [41, 42], [43, 44], [45, 46], [47, 48], [49, 50], [51, 52], [53, 54], [55, 56]]
#
# colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
#           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
#           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def draw_kpts(canvas, kpts):
    # draw points
    num = len(kpts)

    for i in range(num):
        for j in range(len(colors)):
            if kpts[i][j][2] is not 2:
                cv2.circle(canvas,tuple([kpts[i][j][0],kpts[i][j][1]]),4, colors[i], thickness=-1)
    # draw lines
    return canvas


def draw_result(heatmap_avg, paf_avg, canvas):
    heatmap_avg = heatmap_avg.transpose(1, 2, 0)  # 46, 46, 19
    heatmap_avg = cv2.resize(heatmap_avg, (368, 368))

    paf_avg = paf_avg.transpose(1, 2, 0)  # 46, 46, 38
    paf_avg = cv2.resize(paf_avg, (368, 368))

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0

    for part in range(19 - 1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    #score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                    #    0.5 * oriImg.shape[0] / norm - 1, 0)
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts)

                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print "found = 2"
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # draw points
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # draw lines
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def get_pose(heatmap, vecmap):
    '''
    This function aims to get points and lines 
        from heatmap and vecmap.
    size:
        heatmap:(19, 46, 46)
        vecmap: (38, 46, 46)
    return:
        peaks: list of points, 
        subsets: list of pairs
    '''
    heatmap = heatmap.transpose(1, 2, 0)# 46, 46, 19
    heatmap = cv2.resize(heatmap, (368, 368))

    vecmap = vecmap.transpose(1, 2, 0) # 46, 46, 38
    vecmap = cv2.resize(vecmap, (368, 368))
    
    height, width = 368, 368
    all_peaks = []   # all of the possible points by classes.
    peak_counter = 0

    for part in range(1, 19):
        map_ori = heatmap[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # (w, h)
        
        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i], ) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = [] # save all of the possible lines by classes.
    special_k = []      # save the lines, which haven't legal points.
    mid_num = 10        # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = vecmap[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2]) # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1, 0) # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3], reverse=True) # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]]) 
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    return all_peaks, subset, candidate
        

def draw_pose(canvas, peaks, subset, candidate):
    # draw points
    #print('peaks', len(peaks))
    for i in range(18):
        for j in range(len(peaks[i])):
            cv2.circle(canvas, peaks[i][j][0:2], 4, colors[i], thickness=-1)
    hmap = canvas.transpose(2, 0, 1)
    
    # draw lines
    #print('subset', len(subset))
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    #vmap = canvas.transpose(2, 0, 1)
    #return hmap, vmap


def vis_pose(imgs, heats, vecs, masks, num):
    mask_res, heat_res, vec_res = [], [], []
    for i in range(num):
        img = imgs[i,:,:,:] # 3, 368, 368
        img *= 256
        img += 128
        #img *= 128
        #img += 128
        #img /= 255
        
        mask = masks[i,:,:,:]
        mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, (368, 368))
        mask[mask > 1] = 1
        mask = mask.reshape((1, 368, 368))
        m = img * mask
        mask_res.append(m[[2,1,0],:,:])
        
        img = img.transpose(1, 2, 0)# 368, 368, 3
        vec = vecs[i,:,:,:]
        heat = heats[i,:,:,:] 
        
        peaks, subset, candidate = get_pose(heat, vec)
        hmap, vmap = draw_pose(img, peaks, subset, candidate)
        heat_res.append(hmap[[2,1,0],:,:])
        vec_res.append(vmap[[2,1,0],:,:])
    
    return heat_res, vec_res
    
    
def read_json_pose(gt_path, id, img):
    kpts, _, _ = CocoFolder.read_json_file(gt_path)
    kpt = kpts[id]
    print('kpt', kpt)
    for person in kpt:
        for i, point in enumerate(person):
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img, tuple([x, y]), 3, colors[i], thickness=-1)

    return img


def vis_test(img_list, num):
    ### show grount truth
    gt_path = "/data/xiaobing.wang/qy.feng/data/coco_pytorch/test_json_file.json"
    mk_path = "/data/xiaobing.wang/qy.feng/data/coco_pytorch/test_maskmiss_list.txt"
    # im_id, im_list = utils.get_test_id(100)
    mk_list = CocoFolder.read_data_file(mk_path)

    ids = np.random.randint(2644, size=num)
    print('ids',ids)
    gt_list, mask_list, heat_list, vec_list = [], [], [], []
    kpt_list, center_list, scale_list = CocoFolder.read_json_file(gt_path)
    im_paths = []
        
    for id in ids:
        ### show pose from gt
        path = img_list[id]
        im_paths.append(path)
        canvas = cv2.imread(path)
        kpt, center, scale = kpt_list[id], center_list[id], scale_list[id]

        for person in kpt:
            for i, point in enumerate(person):
                x = int(point[0])
                y = int(point[1])
                cv2.circle(canvas, tuple([x, y]), 4, colors[i], thickness=-1)

        gt_list.append(canvas.transpose(2, 0, 1))

        ### show pose from map
        stride = 8
        theta = 1.0
        sigma = 7.0
        p_n, v_n = 19, 19
        vec_pair = [[2,3,5,6,8,9, 11,12,0,1,1, 1,1,2, 5, 0, 0, 14,15],
                    [3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,14,15,16,17]]
        img = cv2.imread(path)
        mask_path = mk_list[id]
        mask = np.load(mask_path)
        mask = np.array(mask, dtype=np.float32)

        transformer = Mytransforms.Compose([Mytransforms.TestResized(368),])
        img, mask, kpt, center = transformer(img, mask, kpt, center, scale)

        height, width, _ = img.shape
        mask = cv2.resize(mask, (width / stride, height / stride)).reshape((height / stride, width / stride, 1))
        
        heatmap = np.zeros((height / stride, width / stride, p_n), dtype=np.float32)
        heatmap = CocoFolder.generate_heatmap(heatmap, kpt, stride, sigma)
        heatmap[:,:,0] = 1.0 - np.max(heatmap[:,:,1:], axis=2) # for background
        heatmap = heatmap * mask

        vecmap = np.zeros((height / stride, width / stride, v_n*2), dtype=np.float32)
        cnt = np.zeros((height / stride, width / stride, v_n), dtype=np.int32)
        vecmap = CocoFolder.generate_vector(vecmap, cnt, kpt, vec_pair, stride, theta)
        vecmap = vecmap * mask

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) # mean, std

        mask_res, heat_res, vec_res = utils.show_ori_pose(img.numpy(), heatmap, vecmap, mask)
        mask_list.append(mask_res.transpose(2, 0, 1))
        heat_list.append(heat_res.transpose(2, 0, 1))
        vec_list.append(vec_res.transpose(2, 0, 1))


    return gt_list, mask_list, heat_list, vec_list, im_paths
