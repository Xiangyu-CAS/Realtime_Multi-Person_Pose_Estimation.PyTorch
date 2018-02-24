import torch
import coco_loader
import CocoFolder
import Mytransforms
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

dir = ['/home/xiangyu/data/samsung_pose_data/img_list/valminusminival2014.txt',
       '/home/xiangyu/data/samsung_pose_data/mask_list/valminusminival2014.txt',
       '/home/xiangyu/data/samsung_pose_data/json/valminusminival2014.json']
out_dir = './vis_input/'

loader = torch.utils.data.DataLoader(
    coco_loader.coco_loader(dir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])),
    batch_size=10, shuffle=False,
    num_workers=1, pin_memory=True)

# dir = ['/home/xiangyu/data/samsung_pose_data/img_list/valminusminival2014.txt',
#        '/home/xiangyu/data/samsung_pose_data/mask_list/valminusminival2014.txt',
#        '/home/xiangyu/data/samsung_pose_data/json/valminusminival2014.json']
# out_dir = './vis_input/'
#
# loader = torch.utils.data.DataLoader(
#     CocoFolder.CocoFolder(dir, 8,
#                 Mytransforms.Compose([Mytransforms.RandomResized(),
#                 Mytransforms.RandomRotate(40),
#                 Mytransforms.RandomCrop(368),
#                 Mytransforms.RandomHorizontalFlip(),
#             ])),
#     batch_size=10, shuffle=False,
#     num_workers=1, pin_memory=True)


for i, (input, heatmap, vecmap, mask,kpt) in enumerate(loader):
    imgs = input.numpy()
    heats = heatmap.numpy()
    vectors = vecmap.numpy()
    masks = mask.numpy()
    break

for i in range(10):
    img = imgs[i, :, :, :]
    img = img.transpose(1, 2, 0)
    img *= 128
    img += 128

    #img /= 255
    # plt.imshow(img)
    # plt.show()
    # plt.close()

    mask = masks[i, :, :, :]
    mask = mask.transpose(1, 2, 0)
    mask = cv2.resize(mask, (368, 368))
    mask = mask.reshape((368, 368, 1))
    new_img = img * mask
    img = np.array(img, np.uint8)
    new_img = np.array(new_img, np.uint8)
    # plt.imshow(new_img)
    # plt.show()
    # plt.close()

    heatmaps = heats[i, :, :, :]
    heatmaps = heatmaps.transpose(1, 2, 0)
    heatmaps = cv2.resize(heatmaps, (368, 368))
    for j in range(0, 19):
        heatmap = heatmaps[:, :, j]
        heatmap = heatmap.reshape((368, 368, 1))
        heatmap *= 255
        heatmap = np.array(heatmap,np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap = heatmap.reshape((368,368,1))
        #heatmap /= 255
        # result = heatmap * 0.4 + img * 0.5
        print j
        # plt.imshow(img)
        # plt.imshow(heatmap, alpha=0.5)
        # plt.show()
        # plt.close()
        heatmap = cv2.addWeighted(new_img,0.5,heatmap,0.5,0)
        cv2.imwrite(out_dir+ '{}_heatmap_{}.jpg'.format(i,j), heatmap)

    vecs = vectors[i, :, :, :]
    vecs = vecs.transpose(1, 2, 0)
    vecs = cv2.resize(vecs, (368, 368))
    for j in range(0, 38, 2):
        vec = np.abs(vecs[:, :, j])
        vec += np.abs(vecs[:, :, j + 1])
        vec[vec > 1] = 1
        vec = vec.reshape((368, 368, 1))
        # vec[vec > 0] = 1
        vec *= 255
        vec = np.array(vec, np.uint8)
        # vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
        # vec = vec.reshape((368, 368))
        #vec /= 255
        print j
        vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
        vec = cv2.addWeighted(new_img, 0.5, vec, 0.5, 0)
        cv2.imwrite(out_dir + '{}_vec_{}.jpg'.format(i, j), vec)
        # plt.imshow(img)
        # # result = vec * 0.4 + img * 0.5
        # plt.imshow(vec, alpha=0.5)
        # plt.show()
        # plt.close()
    print 'done!'