import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

'''
vgg结构参数
数字代表channels数
M代表最大池化
'''
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'D', 'D', 'D'],
    # 这是更改后的vgg16，参考了Attentive Feedback Network for Boundary-Aware Salient Object Detection
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

'''
不同stage提取特征的层数，分别在
conv2-2,conv3-3,conv4-3,conv5-3后的ReLU层处进行提取
'''
extract_layers = {
    # 'D': [8, 15, 22, 29]
    'D': [3, 8, 15, 22]
}

pretrained_model_pathes = {
    # 'vgg16': 'vgg16-397923af.pth',
    'vgg16': './models/backbones/vgg16-397923af.pth',
}


class VGG(nn.Module):
    def __init__(self, features, ex_layers, init_weights=False):
        '''
        VGG类
        :param features: 原始VGG网络去除全连接层后剩余的前面的网络层
        :param ex_layers: 不同stage提取特征的层数
        :param init_weights: 是否进行参数初始化
        '''
        super(VGG, self).__init__()
        self.features = features
        # extra为在features层后加入的三个卷积层
        # self.extra = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        self.ex_layers = ex_layers
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        '''

        :param x:
        :return: conv1-2,conv2-2,conv3-3,conv4-3再加上conv5-3的tuple
        '''
        # outs = OrderedDict()
        outs = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.ex_layers:
                # print(self.features[i])
                # outs['out' + str(self.ex_layers.index(i) + 2)] = x
                outs.append(x)
        # x = self.extra(x)
        # outs['out6'] = x
        outs.append(x)
        return tuple(outs)

    def _initialize_weights(self):
        # 现在默认使用初始化
        pass


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]  # egnet中用的是kernel_size=3
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 原始
        elif v == 'D':
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2)]
            if batch_norm:
                layers += [nn.BatchNorm2d(in_channels)]
            layers += [nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def load_pretrained_model(model, pretrained_model_path):
    '''

    :param model: net网络
    :param pretrained_model_path: 预训练参数路径 e.g. 'vgg16-397923af.pth'
    :return:
    '''
    print('load pretrained model from ', pretrained_model_path)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    '''
    构建VGG网络
    :param arch: e.g. 'vgg16'
    :param cfg: vgg结构设置 e.g. 'D'
    :param batch_norm:
    :param pretrained:
    :param kwargs:
    :return:
    '''
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), extract_layers[cfg], **kwargs)
    if pretrained:
        model = load_pretrained_model(model, pretrained_model_pathes[arch])
    return model


def vgg16(pretrained=False, **kwargs):
    '''
    VGG16网络
    :param pretrained: 是否加载在ImageNet上的预训练参数
    :param kwargs: init_weights
    :return:
    '''
    print('Constract vgg16 network.')
    return _vgg('vgg16', 'D', batch_norm=False, pretrained=pretrained, **kwargs)


def test():
    # 在这里运行需要修改pretrained路径
    # 下面两行输出网络结构以及shape、params
    net = vgg16(pretrained=False)
    summary(net.cuda(), (3, 360, 640))

    # x = torch.ones(1, 3, 224, 224)
    # outs = net(x)
    # for o in outs:
    #     print(o.shape)
    # ----------------print:---------------
    # torch.Size([1, 64, 224, 224])
    # torch.Size([1, 128, 112, 112])
    # torch.Size([1, 256, 56, 56])
    # torch.Size([1, 512, 28, 28])
    # torch.Size([1, 512, 28, 28])


if __name__ == '__main__':
    test()
