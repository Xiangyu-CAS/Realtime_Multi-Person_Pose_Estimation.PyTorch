import argparse
import os
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.append('..')
import torch
import pose_estimation
import cv2
#import model.vgg_1branch as vgg_1branch

# limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], [10,11], [12,13], [13,14], [1,2], [2,9], [2,12], [2,3], [2,6], \
#           [3,17],[6,18],[1,16],[1,15],[16,18],[15,17]]
#
# # limbSeq = [[3,4], [4,5], [6,7],[9,10],[10,11],[12,13],[13,14],[1,2],[2,9],[2,12], [2,3],[2,6], \
# #            [3,17],[6,18],[1,15],[1,16],[15,17],[16,18]]
#
# mapIdx = [[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40], \
#           [41,42],[43,44],[45,46],[47,48],[49,50],[51,52],[53,54],[55,56]]

# find connection in the specified sequence, center 29 is in the position 15
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
thre_point = 0.15
thre_line = 0.05
stickwidth = 4

def construct_model(args):

    model = pose_estimation.PoseModel(num_point=19, num_vector=19)
    #model = vgg_1branch.PoseModel(num_point=19, num_vector=19)

    state_dict = torch.load(args.model)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # state_dict = model.state_dict()
    # state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    return model

def padRightDownCorner(img, stride, padValue):

    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def normalize(origin_img):


    origin_img = np.array(origin_img, dtype=np.float32)
    origin_img -= 128.0
    origin_img /= 256.0

    return origin_img

def process(model, input_path):

    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape

    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 19)) # num_point
    paf_avg = np.zeros((height, width, 38))     # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, stride, padValue)

        input_img = np.transpose(imgToTest_padded[:,:,:,np.newaxis], (3, 2, 0, 1)) # required shape (1, c, h, w)
        mask = np.ones((1, 1, input_img.shape[2] / stride, input_img.shape[3] / stride), dtype=np.float32)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())
        mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())

        # get the features
        vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)

        # get the heatmap
        heatmap = heat6.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec6.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0)) # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []   # all of the possible points by classes.
    peak_counter = 0

    for part in range(0, 18): # 1-19
        map_ori = heatmap_avg[:, :, part]
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
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
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

        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
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

    # draw points
    canvas = cv2.imread(input_path)
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

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='input image',
                        default='ski.jpg')
    parser.add_argument('--output', type=str, default='result.jpg', help='output image')
    parser.add_argument('--model', type=str, help='path to the weights file',
                        default = '/home/xiangyu/data/pretrain/COCO/coco_pose_iter_440000.pth.tar')

    args = parser.parse_args()
    input_image = args.image
    output = args.output

    # load model
    model = construct_model(args)

    tic = time.time()
    print('start processing...')

    # generate image with body parts
    canvas = process(model, input_image)

    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)
