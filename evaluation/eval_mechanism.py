import os
import sys
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage.filters import gaussian_filter
import math, time
import torch
sys.path.append('../')


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



def mechanism(img, img_anns):
    # -----------------------generate GT-------------------------------------
    COCO_TO_OURS = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    vec_pair = [[2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
                [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]

    lists = []

    numPeople = len(img_anns)
    persons = []
    person_centers = []
    for p in range(numPeople):

        if img_anns[p]['num_keypoints'] < 5 or img_anns[p]['area'] < 32 * 32:
            continue
        kpt = img_anns[p]['keypoints']
        dic = dict()

        # person center
        person_center = [img_anns[p]['bbox'][0] + img_anns[p]['bbox'][2] / 2.0,
                         img_anns[p]['bbox'][1] + img_anns[p]['bbox'][3] / 2.0]
        scale = img_anns[p]['bbox'][3] / 368.0

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in person_centers:
            dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0]) + (person_center[1] - pc[1]) * (
                person_center[1] - pc[1]))
            if dis < pc[2] * 0.3:
                flag = 1;
                break
        if flag == 1:
            continue
        dic['objpos'] = person_center
        dic['keypoints'] = np.zeros((17, 3)).tolist()
        dic['scale'] = scale
        for part in range(17):
            dic['keypoints'][part][0] = kpt[part * 3]
            dic['keypoints'][part][1] = kpt[part * 3 + 1]
            # visiable is 1, unvisiable is 0 and not labeled is 2
            if kpt[part * 3 + 2] == 2:
                dic['keypoints'][part][2] = 1
            elif kpt[part * 3 + 2] == 1:
                dic['keypoints'][part][2] = 0
            else:
                dic['keypoints'][part][2] = 2

        transform_dict = dict()
        transform_dict['keypoints'] = np.zeros((18, 3)).tolist()
        for i in range(17):
            transform_dict['keypoints'][COCO_TO_OURS[i]][0] = dic['keypoints'][i][0]
            transform_dict['keypoints'][COCO_TO_OURS[i]][1] = dic['keypoints'][i][1]
            transform_dict['keypoints'][COCO_TO_OURS[i]][2] = dic['keypoints'][i][2]
        transform_dict['keypoints'][1][0] = (dic['keypoints'][5][0] + dic['keypoints'][6][0]) * 0.5
        transform_dict['keypoints'][1][1] = (dic['keypoints'][5][1] + dic['keypoints'][6][1]) * 0.5

        if dic['keypoints'][5][2] == dic['keypoints'][6][2]:
            transform_dict['keypoints'][1][2] = dic['keypoints'][5][2]
        elif dic['keypoints'][5][2] == 2 or dic['keypoints'][6][2] == 2:
            transform_dict['keypoints'][1][2] = 2
        else:
            transform_dict['keypoints'][1][2] = 0

        persons.append(transform_dict)

    kpt = []
    for person in persons:
        kpt.append(person['keypoints'])

    if len(kpt) == 0:
        return [],[],img.copy()

    import coco_loader
    stride = 8
    theta = 1.0
    sigma = 7.0
    height, width, _ = img.shape
    heatmap = np.zeros((height / stride, width / stride, len(kpt[0]) + 1), dtype=np.float32)
    heatmap = coco_loader.generate_heatmap(heatmap, kpt, stride, sigma)
    heatmap[:, :, -1] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)  # for background

    vecmap = np.zeros((height / stride, width / stride, len(vec_pair[0]) * 2), dtype=np.float32)
    cnt = np.zeros((height / stride, width / stride, len(vec_pair[0])), dtype=np.int32)
    vecmap = coco_loader.generate_vector(vecmap, cnt, kpt, vec_pair, stride, theta)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    vecmap = cv2.resize(vecmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # ---------------post processing----------------------------------
    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0

    for part in range(19 - 1):
        x_list = []
        y_list = []
        map_ori = heatmap[:, :, part]
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
        score_mid = vecmap[:, :, [x - 19 for x in mapIdx[k]]]
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

    canvas = img.copy()
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


    return candidate, subset, canvas

def main():
    import pose_estimation

    model = pose_estimation.PoseModel(num_point=19, num_vector=19)

    img_dir = '/home/bst2017/workspace/data/coco/images/val2017/'
    annFile = '/home/bst2017/workspace/data/coco/annotations/person_keypoints_val2017.json'
    num_imgs = 50 # COCO 38%
    orderCOCO = [0, -1, 6, 8, 10, 5, 7, 9,  12, 14, 16, 11, 13, 15, 2, 1, 4, 3] #[1, 0, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    myjsonValidate = list(dict())

    cocoGt = COCO(annFile)
    img_names = cocoGt.imgs
    # filter only person
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    #ids = list(cocoGt.imgs.keys())
    #--------------------------------------------------------
    #
    for i in range(num_imgs):
        print('{}/{}'.format(i,num_imgs))
        img_info = cocoGt.loadImgs(imgIds[i])[0]
        image_id = img_info['id']
        oriImg = cv2.imread(os.path.join(img_dir, img_info['file_name']))
        ann_ids = cocoGt.getAnnIds(imgIds=image_id)
        img_anns = cocoGt.loadAnns(ann_ids)

        candidate, subset,canvas = mechanism(oriImg, img_anns)
        cv2.imwrite(os.path.join('./result', img_info['file_name']), canvas)
        for j in range(len(subset)):
            category_id = 1
            keypoints = np.zeros(51)
            score = 0
            for part in range(18):
                if part == 1:
                    continue
                index = int(subset[j][part])
                if index > 0:
                    #realpart = orderCOCO[part] - 1
                    realpart = orderCOCO[part]
                    if realpart == -1:
                        continue
                    # if part == 0:
                    #     keypoints[realpart * 3] = candidate[index][0] -0.5
                    #     keypoints[realpart * 3 + 1] = candidate[index][1] -0.5
                    #     keypoints[realpart * 3 + 2] = 1
                    #     # score = score + candidate[index][2]
                    else:
                        keypoints[(realpart) * 3] = candidate[index][0]
                        keypoints[(realpart) * 3 + 1] = candidate[index][1]
                        keypoints[(realpart) * 3 + 2] = 1
                        # score = score + candidate[index][2]

            keypoints_list = keypoints.tolist()
            current_dict = {'image_id': image_id,
                            'category_id': category_id,
                            'keypoints': keypoints_list,
                            'score': subset[j][-2]}
            myjsonValidate.append(current_dict)
            #count = count + 1
    import json
    with open('evaluationResult.json', 'w') as outfile:
        json.dump(myjsonValidate, outfile)
    resJsonFile = 'evaluationResult.json'
    cocoDt2 = cocoGt.loadRes(resJsonFile)

    image_ids = []
    for i in range(num_imgs):
        img = cocoGt.loadImgs(imgIds[i])[0]
        image_ids.append(img['id'])
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt2, 'keypoints')
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    k = cocoEval.summarize()

if __name__ == '__main__':
    main()