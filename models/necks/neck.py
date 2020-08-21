import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# [cur_channels, featuree_fuse_upsample_channles(if 1 means no channels change; 0 means no fuse),
# edge_fuse_upsample_channels, conv_kernal_size, conv_padding]

# [是否接受上采样的特征融合， 是否接受上采样的边缘信息融合]
neck_cfgs = {
    'baseline': [],
    'modified_1': [[512, 1, 128, 7, 3], [512, 1, 0, 5, 2], [512, 256, 0, 5, 2],
                   [256, 0, 0, 3, 1], [128, 0, 0, 3, 1]],
    'modified_1_fuse': [[False, False], [True, False], [True, False],
                        [True, False], [False, True]],
}


class Neck(nn.Module):
    def __init__(self, merge_arch, predict_arch):
        '''

        :param merge_arch:
        :param predict_arch:
        '''
        super(Neck, self).__init__()
        self.merge_arch = merge_arch
        self.predict_arch = predict_arch

    def forward(self, features):
        '''

        :param features: ['out2':...,'out3':...,'out4':...,'out5':...,'out6':...] 注：现已修改为同样顺序的list
        :return:
        '''
        outs = self.merge_arch(features)
        predicts = self.predict_arch(outs)

        return outs, predicts


class Merge(nn.Module):
    def __init__(self, cfgs, fuse_cfgs):
        super(Merge, self).__init__()
        self.cfgs = cfgs
        self.fuse_cfgs = fuse_cfgs
        # e.g. modified_1: [[512, 1, 128, 7, 3], [512, 1, 0, 5, 2], [512, 256, 0, 5, 2],
        #                    [256, 0, 0, 3, 1], [128, 0, 0, 3, 1]]
        convs, ups, edges = [], [], []
        for cfg in self.cfgs:
            convs.append(nn.Sequential(nn.Conv2d(cfg[0], cfg[0], kernel_size=cfg[3], stride=1, padding=cfg[4]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(cfg[0], cfg[0], kernel_size=cfg[3], stride=1, padding=cfg[4]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(cfg[0], cfg[0], kernel_size=cfg[3], stride=1, padding=cfg[4]),
                                       nn.ReLU(inplace=True)))
            if cfg[1] > 1:  # 1*1卷积 idx = 2
                ups.append(nn.Sequential(nn.Conv2d(cfg[0], cfg[1], 1, 1, bias=False), nn.ReLU(inplace=True)))
            else:
                ups.append(None)

            if cfg[2] > 0:  # idx = 0
                edges.append(nn.Sequential(nn.Conv2d(cfg[0], cfg[2], 1, 1, bias=False),
                                           nn.ReLU(inplace=True)))
            else:
                edges.append(None)

        self.convs = nn.ModuleList(convs)  # len = 5
        self.ups = nn.ModuleList(ups)  # len = 5, 唯一有意义的在 idx = 2,512->256 channels
        self.edges = nn.ModuleList(edges)  # len = 5, 唯一有意义的在 idx = 0, 512->128 channels

    def forward(self, features):
        '''

        :param features: features: ['out2':...,'out3':...,'out4':...,'out5':...,'out6':...]注：现已修改为同样顺序的list
        :return outs:list, conv6到conv2经过top-down的特征输出
        '''
        # features_list = list(reversed(features.values()))  # 从conv6到conv2的特征list
        features_list = features[::-1]
        prev, edges = None, []
        outs = []
        for i in range(len(features_list)):  # i = 0,1,2,3,4
            cfg = self.cfgs[i]
            # e.g. modified_1: [[512, 1, 128, 7, 3], [512, 1, 0, 5, 2], [512, 256, 0, 5, 2],
            #                   [256, 0, 0, 3, 1], [128, 0, 0, 3, 1]]
            fuse_cfg = self.fuse_cfgs[i]
            # e.g. modified_1_fuse:[[False, False], [True, False], [True, False],[True, False], [False, True]]
            out = features_list[i]
            if fuse_cfg[0]:  # 需要接受上采样特征融合 i = 1,2,3
                # shape = [batch_size, channels, height, width]
                if prev.shape[1] > out.shape[1]:  # i=3 512>256,prev特征需要先进行1*1卷积
                    prev = self.ups[i - 1](prev)
                # 然后进行特征融合，需要进行上采样
                out += F.interpolate(prev, out.shape[2:], mode='bilinear', align_corners=True)
            if fuse_cfg[1]:  # 接受边缘信息融合 i = 4
                for idx, edge in enumerate(edges):  # idx = 0
                    out += F.interpolate(self.edges[idx](edge), out.shape[2:], mode='bilinear', align_corners=True)

            outs.append(self.convs[i](out))

            if cfg[2] > 0:  # 需要将该层特征补充给边缘信息
                edges.append(outs[-1])
            prev = outs[-1]

        return outs


class Predict(nn.Module):
    def __init__(self, cfgs, img_hw):
        super(Predict, self).__init__()
        self.cfgs = cfgs
        self.img_hw = img_hw
        predict_convs = []
        for cfg in self.cfgs:
            # e.g. modified_1: [[512, 1, 128, 7, 3], [512, 1, 0, 5, 2], [512, 256, 0, 5, 2],
            #                   [256, 0, 0, 3, 1], [128, 0, 0, 3, 1]]
            predict_convs.append(nn.Conv2d(cfg[0], 1, kernel_size=3, stride=1, padding=1))

        self.predict_convs = nn.ModuleList(predict_convs)

    def forward(self, features):
        '''

        :param features: 上面Merge类输出的outs, conv6到conv2经过top-down的特征输出
        :param img_hw: 图像的高宽, img.shape[2:]
        :return:
        '''
        predicts = []
        for i, feature in enumerate(features):
            predicts.append(
                F.interpolate(self.predict_convs[i](feature), self.img_hw, mode='bilinear', align_corners=True))
        return predicts


def _neck(img_hw, merge_type):
    merge_arch = Merge(neck_cfgs[merge_type], neck_cfgs[merge_type + '_fuse'])
    predict_arch = Predict(neck_cfgs[merge_type], img_hw)
    model = Neck(merge_arch, predict_arch)
    return model


def neck(img_hw, merge_type='modified_1'):
    print('Constract neck...')
    return _neck(img_hw, merge_type)


def test():
    outs = OrderedDict()
    outs['outs2'] = torch.randn(1, 128, 112, 112)
    outs['outs3'] = torch.randn(1, 256, 56, 56)
    outs['outs4'] = torch.randn(1, 512, 28, 28)
    outs['outs5'] = torch.randn(1, 512, 14, 14)
    outs['outs6'] = torch.randn(1, 512, 7, 7)
    for k, v in outs.items():
        print(k, v.shape)
    x = torch.randn(1, 3, 224, 224)
    neck_net = neck(x.shape[2:])
    outs2, predicts = neck_net(outs)
    for i in outs2:
        print(i.shape)

    for i in predicts:
        print(i.shape)


if __name__ == '__main__':
    test()
