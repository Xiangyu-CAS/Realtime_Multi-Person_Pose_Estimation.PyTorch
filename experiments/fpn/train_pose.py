from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

sys.path.append('../../')
import coco_loader
import Mytransforms
from utils import *
import vis_util
import pose_estimation
#from logger import Logger
import model.fpn as fpn


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters',
                        default='config.yml')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None, type=str,
                        dest='pretrained', help='the path of pretrained model')

    parser.add_argument('--snapshot', type=str,
                        dest='snapshot', help='resume model',
                        #default='/home/xiangyu/samsung_pose/experiments/baseline/40000_val.pth.tar'
                        default=None
                        )

    parser.add_argument('--root', type=str,
                        dest='root', help='the root of images',
                        default='/data/root/data/coco/images/val2014')
    parser.add_argument('--train_dir', nargs='+', type=str,
                        dest='train_dir', help='the path of train file',
                        # default=['/home/xiangyu/data/samsung_pose_data/img_list/valminusminival2014.txt',
                        #          '/home/xiangyu/data/samsung_pose_data/mask_list/valminusminival2014.txt',
                        #          '/home/xiangyu/data/samsung_pose_data/json/valminusminival2014.json']
                        default=['/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/img_list/valminusminival2014.txt',
                                  '/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/mask_list/valminusminival2014.txt',
                                  '/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/json/valminusminival2014.json']
                        )
    parser.add_argument('--val_dir', nargs='+', type=str,
                        dest='val_dir', help='the path of val file',
                        default=['/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/img_list/minival2014.txt',
                                 '/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/mask_list/minival2014.txt',
                                 '/home/xiangyuzhu/workspace/data/coco/samsung_pose_data/json/minival2014.json'])
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')
    parser.add_argument('--logdir', default='./logs', type=str,
                        dest='logdir', help='path of log')
    return parser.parse_args()


def construct_model(args):
    if not args.snapshot:
        resnet_model = '/home/xiangyuzhu/workspace/data/pretrain/ResNet/resnet50-19c8e357.pth'
        print('--------load pretrain model from {}----------------'.format(resnet_model))
        model = fpn.Pose_Estimation(vec_num=38, heat_num=19)
        #model.load_weights(resnet_model)
    else:
        print('--------load snapshot from {}----------------'.format(args.snapshot))
        model = fpn.Pose_Estimation(vec_num=38, heat_num=19)
        state_dict = torch.load(args.snapshot)['state_dict']
        model.load_state_dict(state_dict)

    print(model)
    model.cuda()

    return model


def get_parameters(model, config, isdefault=True):
    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
              {'params': lr_2, 'lr': config.base_lr * 2.},
              {'params': lr_4, 'lr': config.base_lr * 4.},
              {'params': lr_8, 'lr': config.base_lr * 8.}]

    return params, [1., 2., 4., 8.]


def to_np(x):
    return x.data.cpu().numpy()


def train_val(model, args):
    traindir = args.train_dir
    valdir = args.val_dir

    config = Config(args.config)
    cudnn.benchmark = True

    # Set the logger
    #logger = Logger('./log')

    train_loader = torch.utils.data.DataLoader(
        coco_loader.coco_loader(traindir, 8,
                              Mytransforms.Compose([Mytransforms.RandomResized(),
                                                    Mytransforms.RandomRotate(40),
                                                    Mytransforms.RandomCrop(368),
                                                    Mytransforms.RandomHorizontalFlip(),
                                                    ])),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)

    # if config.test_interval != 0 and args.val_dir is not None:
    #     val_loader = torch.utils.data.DataLoader(
    #         coco_loader.coco_loader(valdir, 8,
    #                               Mytransforms.Compose([Mytransforms.TestResized(368),
    #                                                     ])),
    #         batch_size=4, shuffle=False,
    #         num_workers=config.workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()

    params, multiple = get_parameters(model, config, False)

    optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(12)]
    top1 = AverageMeter()
    topk = AverageMeter()

    end = time.time()
    iters = config.start_iters
    best_model = config.best_model
    learning_rate = config.base_lr

    model.train()

    heat_weight = 46 * 46 * 19 / 2.0  # for convenient to compare with origin code
    vec_weight = 46 * 46 * 38 / 2.0

    while iters < config.max_iter:
        # ---------------------------------------------------- train ------------------------------------------
        for i, (input, heatmap, vecmap, mask, kpt) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy,
                                                 policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            input = input.cuda(async=True)
            heatmap = heatmap.cuda(async=True)
            vecmap = vecmap.cuda(async=True)
            mask = mask.cuda(async=True)

            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            vecmap_var = torch.autograd.Variable(vecmap)
            mask_var = torch.autograd.Variable(mask)

            #vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)
            vec1, heat1, vec2, heat2 = model(input_var, mask_var)
            loss1_1 = criterion(vec1, vecmap_var) * vec_weight
            loss1_2 = criterion(heat1, heatmap_var) * heat_weight
            loss2_1 = criterion(vec2, vecmap_var) * vec_weight
            loss2_2 = criterion(heat2, heatmap_var) * heat_weight
            # loss3_1 = criterion(vec3, vecmap_var) * vec_weight
            # loss3_2 = criterion(heat3, heatmap_var) * heat_weight
            # loss4_1 = criterion(vec4, vecmap_var) * vec_weight
            # loss4_2 = criterion(heat4, heatmap_var) * heat_weight
            # loss5_1 = criterion(vec5, vecmap_var) * vec_weight
            # loss5_2 = criterion(heat5, heatmap_var) * heat_weight
            # loss6_1 = criterion(vec6, vecmap_var) * vec_weight
            # loss6_2 = criterion(heat6, heatmap_var) * heat_weight

            #loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2
            loss = loss1_1 + loss1_2 + loss2_1 + loss2_2

            losses.update(loss.data[0], input.size(0))
            # loss_list = [loss1_1, loss1_2, loss2_1, loss2_2, loss3_1, loss3_2, loss4_1, loss4_2, loss5_1, loss5_2,
            #              loss6_1, loss6_2]
            loss_list = [loss1_1, loss1_2, loss2_1, loss2_2]
            for cnt, l in enumerate(loss_list):
                losses_list[cnt].update(l.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                      'Learning rate = {2}\n'
                      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0, 12, 2):
                    print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                          'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1,
                                                                                       loss1=losses_list[cnt],
                                                                                       loss2=losses_list[cnt + 1]))
                print(time.strftime(
                    '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                    time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(12):
                    losses_list[cnt].reset()

            # ------------------------------------------ val ---------------------------------------------------------------------
            # #if config.test_interval != 0 and args.val_dir is not None and iters % config.test_interval == 0:
            # if True:
            #     model.eval()
            #     for j, (input, heatmap, vecmap, mask, kpt) in enumerate(val_loader):
            #         imgs = input.numpy()
            #         heatmap = heatmap.numpy()
            #         vecmap = vecmap.numpy()
            #         mask = mask.numpy()
            #
            #         canvas_targs = np.zeros(imgs.shape)
            #         canvas_preds = np.zeros(imgs.shape)
            #
            #         for i in range(len(imgs)):
            #             img = imgs[i]
            #             img = img.transpose(1, 2, 0)  # 368, 368, 3
            #             img = (img + 1) / 2 * 255
            #
            #             # visualize GT by kpts
            #             # canvas_kpts = img.copy()
            #             # vis_util.draw_kpts(canvas_kpts, kpts)
            #
            #             # visualize results derived from target
            #             canvas_targ = img.copy()
            #             canvas_targ = vis_util.draw_result(heatmap[i], vecmap[i], canvas_targ)
            #             canvas_targ = canvas_targ.transpose(2, 0, 1)
            #             canvas_targs[i] = canvas_targ
            #
            #         # visualize predicted results
            #         input = input.cuda(async=True)
            #         input_var = torch.autograd.Variable(input, volatile=True)
            #         mask_white = np.ones((mask.shape), dtype=np.float32)
            #         mask_white_var = torch.autograd.Variable(torch.from_numpy(mask_white).cuda())
            #
            #         vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 \
            #                                                      = model(input_var,mask_white_var)
            #
            #         heat_out = heat6.data.cpu().numpy()
            #         vec_out = vec6.data.cpu().numpy()
            #
            #         for i in range(len(imgs)):
            #             img = imgs[i]
            #             img = img.transpose(1, 2, 0)  # 368, 368, 3
            #             img = (img + 1) / 2 * 255
            #
            #             canvas_pred = img.copy()
            #             canvas_pred = vis_util.draw_result(heat_out[i], vec_out[i], canvas_pred)
            #             canvas_pred = canvas_pred.transpose(2, 0, 1)
            #             canvas_preds[i] = canvas_pred
            #
            #         ## Log images
            #
            #         imgs = {
            #             'target': canvas_targs,
            #             'predict': canvas_preds
            #         }
            #         for tag, images in imgs.items():
            #             logger.image_summary(tag, images, 0)
            #
            #         #break
            #
            #
            #     model.train()

            if iters % 5000 == 0:
                torch.save({
                    'iter': iters,
                    'state_dict': model.state_dict(),
                }, str(iters) + '.pth.tar')

            if iters == config.max_iter:
                break


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse()
    model = construct_model(args)
    train_val(model, args)