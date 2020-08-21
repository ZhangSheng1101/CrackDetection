import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=1),
            BasicConv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class ConcatOutput_S(nn.Module):
    def __init__(self, channels):
        super(ConcatOutput_S, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv2 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv3 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv4 = BasicConv2d(channels, channels, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )
        self.conv_cat4 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x4 = torch.cat((x4, self.conv1(x5)), 1)  # [batch_size,channels,h,w]
        x4 = self.conv_cat1(x4)

        x3 = torch.cat((x3, self.conv2(self.upsample(x4))), 1)
        x3 = self.conv_cat2(x3)

        x2 = torch.cat((x2, self.conv3(self.upsample(x3))), 1)
        x2 = self.conv_cat3(x2)

        x1 = torch.cat((x1, self.conv4(self.upsample(x2))), 1)
        x1 = self.conv_cat4(x1)

        return self.output(x1)


class ConcatOutput_E(nn.Module):
    def __init__(self, channels):
        super(ConcatOutput_E, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv2 = BasicConv2d(channels, channels, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channels, 2 * channels, 3, padding=1),
            BasicConv2d(2 * channels, channels, 1)
        )

        self.output = nn.Sequential(
            BasicConv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x1, x2, x3):
        x2 = torch.cat((x2, self.conv1(self.upsample(x3))), 1)
        x2 = self.conv_cat1(x2)

        x1 = torch.cat((x1, self.conv2(self.upsample(x2))), 1)
        x1 = self.conv_cat2(x1)

        return self.output(x1)


class Head_FPN(nn.Module):
    def __init__(self, img_hw, channel=32):
        super(Head_FPN, self).__init__()

        self.img_hw = img_hw

        self.reduce_s1 = Reduction(64, channel)
        self.reduce_s2 = Reduction(128, channel)
        self.reduce_s3 = Reduction(256, channel)
        self.reduce_s4 = Reduction(512, channel)
        self.reduce_s5 = Reduction(512, channel)

        self.reduce_e1 = Reduction(64, channel)
        self.reduce_e2 = Reduction(128, channel)
        self.reduce_e3 = Reduction(256, channel)

        self.ouuput_s = ConcatOutput_S(channel)
        self.ouuput_e = ConcatOutput_E(channel)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features

        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)
        x_s5 = self.reduce_s5(x5)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)

        pred_s = self.ouuput_s(x_s1, x_s2, x_s3, x_s4, x_s5)
        pred_e = self.ouuput_e(x_e1, x_e2, x_e3)

        pred_s = F.upsample(pred_s, size=self.img_hw, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=self.img_hw, mode='bilinear', align_corners=True)

        return pred_s, pred_e


def head_fpn(img_hw):
    print('COnstract head_fpn')
    return Head_FPN(img_hw)
