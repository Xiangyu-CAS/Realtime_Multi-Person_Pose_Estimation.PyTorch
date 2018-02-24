import torch
import torch.nn as nn
import os
import sys
import math
import torchvision.models as models

def make_net_dict():

    feature = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1': [2, 2, 0]},
            {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2': [2, 2, 0]},
            {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]}, {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3': [2, 2, 0]},
            {'conv4_1': [256, 512, 3, 1, 1]}, {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_cpm': [512, 256, 3, 1, 1]}, {'conv4_4_cpm': [256, 128, 3, 1, 1]}]


    block1 = [{'conv5_1_CPM': [128, 128, 3, 1, 1]},{'conv5_2_CPM': [128, 128, 3, 1, 1]},{'conv5_3_CPM': [128, 128, 3, 1, 1]}]#,#,
              #{'conv5_4_CPM': [128, 512, 1, 1, 1]}]


    block2 = [{'Mconv1': [128+38+19, 128, 7, 1, 3]}, {'Mconv2': [128, 128, 7, 1, 3]},
              {'Mconv3': [128, 128, 7, 1, 3]},{'Mconv4': [128, 128, 7, 1, 3]},
              {'Mconv5': [128, 128, 7, 1, 3]},
              {'Mconv6': [128, 128, 1, 1, 0]}
              ]

    predict_layers = [[{'predict_L1': [128, 38, 1, 1, 0]}],
                      [{'predict_L2': [128, 19, 1, 1, 0]}]]

    net_dict = [feature,block1,predict_layers,block2,predict_layers]

    return net_dict


class vgg_1branch(nn.Module):

    def __init__(self, net_dict, batch_norm=False):

        super(vgg_1branch, self).__init__()

        self.feature = self._make_layer(net_dict[0])

        self.block1 = self._make_layer(net_dict[1])

        self.predict_L1_stage1 = self._make_layer(net_dict[2][0])
        self.predict_L2_stage1 = self._make_layer(net_dict[2][1])

        # repeate
        self.block2 = self._make_layer(net_dict[3])

        self.predict_L1_stage2 = self._make_layer(net_dict[4][0])
        self.predict_L2_stage2 = self._make_layer(net_dict[4][1])

        self.block3 = self._make_layer(net_dict[3])

        self.predict_L1_stage3 = self._make_layer(net_dict[4][0])
        self.predict_L2_stage3 = self._make_layer(net_dict[4][1])

        self.block4 = self._make_layer(net_dict[3])

        self.predict_L1_stage4 = self._make_layer(net_dict[4][0])
        self.predict_L2_stage4 = self._make_layer(net_dict[4][1])

        self.block5 = self._make_layer(net_dict[3])

        self.predict_L1_stage5 = self._make_layer(net_dict[4][0])
        self.predict_L2_stage5 = self._make_layer(net_dict[4][1])

        self.block6 = self._make_layer(net_dict[3])

        self.predict_L1_stage6 = self._make_layer(net_dict[4][0])
        self.predict_L2_stage6 = self._make_layer(net_dict[4][1])


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, net_dict, batch_norm=False):
        layers = []
        length = len(net_dict)
        for i in range(length):
            one_layer = net_dict[i]
            key = one_layer.keys()[0]
            v = one_layer[key]

            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x, mask):
        # define forward flow
        feature = self.feature(x)

        out_stage1 = self.block1(feature)
        L1_stage1 = self.predict_L1_stage1(out_stage1)
        L2_stage1 = self.predict_L2_stage1(out_stage1)
        L1_stage1_mask = L1_stage1 * mask
        L2_stage1_mask = L2_stage1 * mask

        concat_stage2 = torch.cat([L1_stage1, L2_stage1, feature], 1)
        out_stage2 = self.block2(concat_stage2)
        L1_stage2 = self.predict_L1_stage2(out_stage2)
        L2_stage2 = self.predict_L2_stage2(out_stage2)
        L1_stage2_mask = L1_stage2 * mask
        L2_stage2_mask = L2_stage2 * mask

        concat_stage3 = torch.cat([L1_stage2, L2_stage2, feature], 1)
        out_stage3 = self.block3(concat_stage3)
        L1_stage3 = self.predict_L1_stage3(out_stage3)
        L2_stage3 = self.predict_L2_stage3(out_stage3)
        L1_stage3_mask = L1_stage3 * mask
        L2_stage3_mask = L2_stage3 * mask

        concat_stage4 = torch.cat([L1_stage3, L2_stage3, feature], 1)
        out_stage4 = self.block4(concat_stage4)
        L1_stage4 = self.predict_L1_stage4(out_stage4)
        L2_stage4 = self.predict_L2_stage4(out_stage4)
        L1_stage4_mask = L1_stage4 * mask
        L2_stage4_mask = L2_stage4 * mask

        concat_stage5 = torch.cat([L1_stage4, L2_stage4, feature], 1)
        out_stage5 = self.block5(concat_stage5)
        L1_stage5 = self.predict_L1_stage5(out_stage5)
        L2_stage5 = self.predict_L2_stage5(out_stage5)
        L1_stage5_mask = L1_stage5 * mask
        L2_stage5_mask = L2_stage5 * mask

        concat_stage6 = torch.cat([L1_stage5, L2_stage5, feature], 1)
        out_stage6 = self.block6(concat_stage6)
        L1_stage6 = self.predict_L1_stage6(out_stage6)
        L2_stage6 = self.predict_L2_stage6(out_stage6)
        L1_stage6_mask = L1_stage6 * mask
        L2_stage6_mask = L2_stage6 * mask

        return L1_stage1_mask,L2_stage1_mask, \
                L1_stage2_mask, L2_stage2_mask, \
                L1_stage3_mask, L2_stage3_mask, \
                L1_stage4_mask, L2_stage4_mask, \
                L1_stage5_mask, L2_stage5_mask, \
                L1_stage6_mask, L2_stage6_mask

def PoseModel(num_point, num_vector, num_stages=6, batch_norm=False, pretrained=False):
    net_dict = make_net_dict()
    model = vgg_1branch(net_dict, batch_norm)

    if pretrained:
        parameter_num = 10
        if batch_norm:
            vgg19 = models.vgg19_bn(pretrained=True)
            parameter_num *= 6
        else:
            vgg19 = models.vgg19(pretrained=True)
            parameter_num *= 2

        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = vgg19_state_dict.keys()

        model_dict = model.state_dict()
        from collections import OrderedDict
        weights_load = OrderedDict()

        for i in range(parameter_num):
            weights_load[model.state_dict().keys()[i]] = vgg19_state_dict[vgg19_keys[i]]
        model_dict.update(weights_load)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    print PoseModel(19, 6, True, True)
