import torch
import torch.nn as nn
import torch.nn.functional as F

head_cfgs = {
    'baseline': [[128], [256, 512, 512, 512]],
    'baseline_conv': [[3, 1], [5, 2], [5, 2], [7, 3]],
}


class Head(nn.Module):
    def __init__(self, cfgs, conv_cfgs, ff_arch, img_hw):
        super(Head, self).__init__()
        self.cfgs = cfgs
        self.conv_cfgs = conv_cfgs
        self.ff_arch = ff_arch
        self.img_hw = img_hw

        convs, predicts = [], []
        for i, cfg in enumerate(self.cfgs[1]):
            convs.append(nn.Sequential(
                nn.Conv2d(self.cfgs[0][0], self.cfgs[0][0], self.conv_cfgs[i][0], 1, self.conv_cfgs[i][1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfgs[0][0], self.cfgs[0][0], self.conv_cfgs[i][0], 1, self.conv_cfgs[i][1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfgs[0][0], self.cfgs[0][0], self.conv_cfgs[i][0], 1, self.conv_cfgs[i][1]),
                nn.ReLU(inplace=True)))
            predicts.append(nn.Sequential(nn.Conv2d(self.cfgs[0][0], 1, 3, 1, 1)))

        self.convs = nn.ModuleList(convs)  # conv3-6的黄色卷积
        self.predicts = nn.ModuleList(predicts)
        self.final_predict = nn.Sequential(nn.Conv2d(self.cfgs[0][0], self.cfgs[0][0], 5, 1, 2), nn.ReLU(inplace=True),
                                           nn.Conv2d(self.cfgs[0][0], 1, 3, 1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        '''

        :param features: list, conv6到conv2经过top-down的特征输出
        :return:
        '''
        predicts = []
        temp_feat = []
        outs = self.ff_arch(features)  # conv3-6经过FF后的特征, shape = [1,128,112,112]
        for idx, out in enumerate(outs):
            out = self.convs[idx](out)
            temp_feat.append(out)
            predicts.append(F.interpolate(self.predicts[idx](out), self.img_hw, mode='bilinear', align_corners=True))

        final_predict = temp_feat[0]
        for idx in range(1, len(temp_feat)):
            final_predict = self.relu(torch.add(final_predict, temp_feat[idx]))
        final_predict = F.interpolate(self.final_predict(final_predict), self.img_hw, mode='bilinear',
                                      align_corners=True)
        return predicts, final_predict


class FF(nn.Module):
    def __init__(self, cfgs):
        super(FF, self).__init__()
        self.cfgs = cfgs
        # [[128], [256, 512, 512, 512]]
        ups = []
        for idx, salient_channels in enumerate(self.cfgs[1]):
            part_ups = []
            for edge_channels in self.cfgs[0]:
                part_ups.append(
                    nn.Sequential(nn.Conv2d(salient_channels, edge_channels, 1, 1, bias=False), nn.ReLU(inplace=True)))
            ups.append(nn.ModuleList(part_ups))

        self.ups = nn.ModuleList(ups)  # [[256->128],[512->128],[512->128],[512->128]]

    def forward(self, features):
        '''

        :param features:list, conv6到conv2经过top-down的特征输出
        :return: outs:conv3-6经过FF后的特征
        '''
        outs = []
        features = features[::-1]
        edge_nums = len(self.cfgs[0])
        edges = features[:edge_nums]
        salients = features[edge_nums:]

        for i, salient in enumerate(salients):  # [1,256,56,56],[1,512,28,28],[1,512,14,14],[1,512,7,7]
            for j, edge in enumerate(edges):  # [1,128,112,112]
                fuse_feat = self.ups[i][j](salient)  # [1,128,56,56],[1,128,28,28],[1,128,14,14],[1,128,7,7]
                fuse_feat = F.interpolate(fuse_feat, edge.shape[2:], mode='bilinear',
                                          align_corners=True)  # [1,128,112,112]
                fuse_feat += edge
                outs.append(fuse_feat)

        return outs


def _head(img_hw, merge_type):
    ff_arch = FF(head_cfgs[merge_type])
    model = Head(head_cfgs[merge_type], head_cfgs[merge_type + '_conv'], ff_arch, img_hw)
    return model


def head(img_hw, merge_type='baseline'):
    print('Constract head...')
    return _head(img_hw, merge_type)


def test():
    features = []
    features.append(torch.randn(1, 512, 7, 7))
    features.append(torch.randn(1, 512, 14, 14))
    features.append(torch.randn(1, 512, 28, 28))
    features.append(torch.randn(1, 256, 56, 56))
    features.append(torch.randn(1, 128, 112, 112))

    x = torch.randn(1, 3, 224, 224)
    head_net = head(x.shape[2:])
    predicts, final_predict = head_net(features)

    for p in predicts:
        print(p.shape)

    print(final_predict.shape)


if __name__ == '__main__':
    test()
