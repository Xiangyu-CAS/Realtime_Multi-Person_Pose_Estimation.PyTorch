import os
import sys
import math
import json
import numpy as np
from pycocotools.coco import COCO

'''
coco_annotations

   u'keypoints': [
    0.u'nose',
    1.u'left_eye',
    2.u'right_eye',
    3.u'left_ear',
    4.u'right_ear',
    5.u'left_shoulder',
    6.u'right_shoulder',
    7.u'left_elbow',
    8.u'right_elbow',
    9.u'left_wrist',
    10.u'right_wrist',
    11.u'left_hip',
    12.u'right_hip',
    13.u'left_knee',
    14.u'right_knee',
    15.u'left_ankle',
    16.u'right_ankle'],
    
    
OUR annotations
   u'keypoints': [
    0.u'nose',			->   nose
    1.u'left_eye',		 neck 
    2.u'right_eye',     right_shoulder
    3.u'left_ear',      right_elbow
    4.u'right_ear', right_wrist
    5.u'left_shoulder', left_shoulder
    6.u'right_shoulder', left_elbow
    7.u'left_elbow',  left_wrist
    8.u'right_elbow', right_hip
    9.u'left_wrist', right_knee
    10.u'right_wrist', right_ankle
    11.u'left_hip', left_hip
    12.u'right_hip', left_knee
    13.u'left_knee', left_ankle
    14.u'right_knee',  right_eye
    15.u'left_ankle',  left_eye
    16.u'right_ankle'  right_ear
    17.					 left_ear],
    
'''

def generate_json_mask(ann_path, json_path, mask_dir, filelist_path, masklist_path):
	COCO_Order =   [0, 1,   2,  3,  4, 5, 6, 7, 8, 9, 10,11,12, 13,14, 15, 16]
	COCO_TO_OURS = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

	coco = COCO(ann_path)
	ids = list(coco.imgs.keys())
	lists = []

	filelist_fp = open(filelist_path, 'w')
	masklist_fp = open(masklist_path, 'w')
	for i, img_id in enumerate(ids):
		ann_ids = coco.getAnnIds(imgIds=img_id)
		img_anns = coco.loadAnns(ann_ids)

		numPeople = len(img_anns)
		name = coco.imgs[img_id]['file_name']
		height = coco.imgs[img_id]['height']
		width = coco.imgs[img_id]['width']

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

			persons.append(dic)
			person_centers.append(np.append(person_center, max(img_anns[p]['bbox'][2], img_anns[p]['bbox'][3])))

		if len(persons) > 0:
			filelist_fp.write(name + '\n')
			info = dict()
			info['filename'] = name
			info['info'] = []
			cnt = 1
			for person in persons:
				dic = dict()
				dic['pos'] = person['objpos']
				dic['keypoints'] = np.zeros((18, 3)).tolist()
				dic['scale'] = person['scale']
				for i in range(17):
					dic['keypoints'][COCO_TO_OURS[i]][0] = person['keypoints'][i][0]
					dic['keypoints'][COCO_TO_OURS[i]][1] = person['keypoints'][i][1]
					dic['keypoints'][COCO_TO_OURS[i]][2] = person['keypoints'][i][2]
				dic['keypoints'][1][0] = (person['keypoints'][5][0] + person['keypoints'][6][0]) * 0.5
				dic['keypoints'][1][1] = (person['keypoints'][5][1] + person['keypoints'][6][1]) * 0.5
				if person['keypoints'][5][2] == person['keypoints'][6][2]:
					dic['keypoints'][1][2] = person['keypoints'][5][2]
				elif person['keypoints'][5][2] == 2 or person['keypoints'][6][2] == 2:
					dic['keypoints'][1][2] = 2
				else:
					dic['keypoints'][1][2] = 0
				info['info'].append(dic)
			lists.append(info)

			mask_all = np.zeros((height, width), dtype=np.uint8)
			mask_miss = np.zeros((height, width), dtype=np.uint8)
			flag = 0
			for p in img_anns:
				if p['iscrowd'] == 1:
					mask_crowd = coco.annToMask(p)
					temp = np.bitwise_and(mask_all, mask_crowd)
					mask_crowd = mask_crowd - temp
					flag += 1
					continue
				else:
					mask = coco.annToMask(p)

				mask_all = np.bitwise_or(mask, mask_all)

				if p['num_keypoints'] <= 0:
					mask_miss = np.bitwise_or(mask, mask_miss)

			if flag < 1:
				mask_miss = np.logical_not(mask_miss)
			elif flag == 1:
				mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
				mask_all = np.bitwise_or(mask_all, mask_crowd)
			else:
				raise Exception('crowd segments > 1')
			np.save(os.path.join(mask_dir, name.split('.')[0] + '.npy'), mask_miss)
			masklist_fp.write(os.path.join(mask_dir, name.split('.')[0] + '.npy') + '\n')
		if i % 1000 == 0:
			print "Processed {} of {}".format(i, len(ids))

	masklist_fp.close()
	filelist_fp.close()
	print 'write json file'

	fp = open(json_path, 'w')
	fp.write(json.dumps(lists))
	fp.close()

	print 'done!'



if __name__ == '__main__':

	# ann_dir = '/home/xiangyu/data/coco/annotations/'
	# img_dir = '/home/xiangyu/data/coco/images/val2014/'
	# out_dir = '/home/xiangyu/data/samsung_pose_data/'
    #
	# # IN
	# train_ann_path = os.path.join(ann_dir, 'person_keypoints_valminusminival2014.json')
	# val_ann_path = os.path.join(ann_dir, 'person_keypoints_minival2014.json')
	# # OUT
	# mask_dir = os.path.join(out_dir, 'mask')
	# json_dir = os.path.join(out_dir, 'json')
	# img_list_dir = os.path.join(out_dir, 'img_list')
	# mask_list_dir = os.path.join(out_dir, 'mask_list')
    #
	# train_json_path = os.path.join(json_dir, 'valminusminival2014.json')
	# val_json_path = os.path.join(json_dir, 'minival2014.json')
    #
	# train_img_path = os.path.join(img_list_dir, 'valminusminival2014.txt')
	# val_img_path = os.path.join(img_list_dir, 'minival2014.txt')
    #
	# train_mask_path = os.path.join(mask_list_dir, 'valminusminival2014.txt')
	# val_mask_path = os.path.join(mask_list_dir, 'minival2014.txt')


#---------------------------------------------------------------------------
	ann_dir = '/home/xiangyu/data/coco/annotations/'
	img_dir = '/home/xiangyu/data/coco/images/train2014/'
	out_dir = '/home/xiangyu/data/samsung_pose_data_train/'

	# IN
	train_ann_path = os.path.join(ann_dir, 'person_keypoints_train2014.json')

	# OUT
	mask_dir = os.path.join(out_dir, 'mask')
	json_dir = os.path.join(out_dir, 'json')
	img_list_dir = os.path.join(out_dir, 'img_list')
	mask_list_dir = os.path.join(out_dir, 'mask_list')

	train_json_path = os.path.join(json_dir, 'train2014.json')

	train_img_path = os.path.join(img_list_dir, 'train2014.txt')

	train_mask_path = os.path.join(mask_list_dir, 'train2014.txt')

	generate_json_mask(train_ann_path, train_json_path, mask_dir, train_img_path, train_mask_path)
	#generate_json_mask(val_ann_path, val_json_path, mask_dir, val_img_path, val_mask_path)
