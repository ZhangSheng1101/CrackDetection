import torch.nn as nn
from .BasicModule import BasicModule
from .backbones import resnet18, resnet50
import torch.nn.functional as F
import torch


# class Net(BasicModule):
#     def __init__(self, pretrained, img_size):
#         super(Net, self).__init__()
#         self.model_name = 'net'
#         self.backbone = vgg16(pretrained=pretrained)
#         self.neck = neck(img_size)
#         self.head = head(img_size)
#
#     def forward(self, x):
#         outs = self.backbone(x)
#         outs, neck_predicts = self.neck(outs)
#         head_predicts, final_predict = self.head(outs)
#
#         return neck_predicts, head_predicts, final_predict
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class TwoBasicConv2d(nn.Module):
    def __init__(self, channel):
        super(TwoBasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x


class DF(nn.Module):
    def __init__(self, channel):
        super(DF, self).__init__()
        # self.conv1 = Conv_upsample(channel)
        # self.conv2 = Conv_upsample(channel)
        # self.conv3 = Conv_upsample(channel)
        # self.conv4 = Conv_upsample(channel)
        # self.conv5 = Conv_upsample(channel)
        # self.conv6 = Conv_upsample(channel)
        # self.conv7 = Conv_upsample(channel)
        # self.conv8 = Conv_upsample(channel)
        # self.conv9 = Conv_upsample(channel)
        # self.conv10 = Conv_upsample(channel)
        # self.conv11 = Conv_upsample(channel)
        # self.conv12 = Conv_upsample(channel)

        self.conv_s1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.fill_1 = TwoBasicConv2d(channel)
        self.fill_2 = TwoBasicConv2d(channel)
        self.fill_3 = TwoBasicConv2d(channel)
        self.fill_4 = TwoBasicConv2d(channel)

        self.conv_e1 = TwoBasicConv2d(channel)
        self.conv_e2 = TwoBasicConv2d(channel)
        self.conv_e3 = TwoBasicConv2d(channel)
        self.conv_e4 = TwoBasicConv2d(channel)

        self.extra_1 = TwoBasicConv2d(channel)
        self.extra_2 = TwoBasicConv2d(channel)
        self.extra_3 = TwoBasicConv2d(channel)
        self.extra_4 = TwoBasicConv2d(channel)

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.fill_1(x_e1) + self.conv_s1(torch.cat((x_s1, x_e1,
                                                                   # self.conv1(x_e2, x_s1),
                                                                   # self.conv2(x_e3, x_s1),
                                                                   # self.conv3(x_e4, x_s1)
                                                                   ), 1))
        x_sf2 = x_s2 + self.fill_2(x_e2) + self.conv_s2(torch.cat((x_s2, x_e2,
                                                                   # self.conv4(x_e3, x_s2),
                                                                   # self.conv5(x_e4, x_s2)
                                                                   ), 1))
        x_sf3 = x_s3 + self.fill_3(x_e3) + self.conv_s3(torch.cat((x_s3, x_e3,
                                                                   # self.conv6(x_e4, x_s3)
                                                                   ), 1))
        x_sf4 = x_s4 + self.fill_4(x_e4) + self.conv_s4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.extra_1(x_s1) + self.conv_e1(x_e1 * x_s1  # *
                                                         # self.conv7(x_s2, x_e1) *
                                                         # self.conv8(x_s3, x_e1) *
                                                         # self.conv9(x_s4, x_e1)
                                                         )
        x_ef2 = x_e2 + self.extra_2(x_s2) + self.conv_e2(x_e2 * x_s2  # *
                                                         # self.conv10(x_s3, x_e2) *
                                                         # self.conv11(x_s4, x_e2)
                                                         )
        x_ef3 = x_e3 + self.extra_3(x_s3) + self.conv_e3(x_e3 * x_s3  # *
                                                         # self.conv12(x_s4, x_e3)
                                                         )
        x_ef4 = x_e4 + self.extra_4(x_s4) + self.conv_e4(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class Conv_upsample(nn.Module):
    def __init__(self, channel):
        super(Conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class DenseFusion(nn.Module):
    def __init__(self, channel):
        super(DenseFusion, self).__init__()
        self.conv1 = Conv_upsample(channel)
        self.conv2 = Conv_upsample(channel)
        self.conv3 = Conv_upsample(channel)
        self.conv4 = Conv_upsample(channel)
        self.conv5 = Conv_upsample(channel)
        self.conv6 = Conv_upsample(channel)
        self.conv7 = Conv_upsample(channel)
        self.conv8 = Conv_upsample(channel)
        self.conv9 = Conv_upsample(channel)
        self.conv10 = Conv_upsample(channel)
        self.conv11 = Conv_upsample(channel)
        self.conv12 = Conv_upsample(channel)

        self.conv_s1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_s4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_e1 = TwoBasicConv2d(channel)
        self.conv_e2 = TwoBasicConv2d(channel)
        self.conv_e3 = TwoBasicConv2d(channel)
        self.conv_e4 = TwoBasicConv2d(channel)

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.conv_s1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_s2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_s3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_s4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_e1(x_e1 * x_s1 *
                                    self.conv7(x_s2, x_e1) *
                                    self.conv8(x_s3, x_e1) *
                                    self.conv9(x_s4, x_e1))
        x_ef2 = x_e2 + self.conv_e2(x_e2 * x_s2 *
                                    self.conv10(x_s3, x_e2) *
                                    self.conv11(x_s4, x_e2))
        x_ef3 = x_e3 + self.conv_e3(x_e3 * x_s3 *
                                    self.conv12(x_s4, x_e3))
        x_ef4 = x_e4 + self.conv_e4(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class BaseNet(BasicModule):
    def __init__(self, depth=18, pretrained=True, channel=32):
        super(BaseNet, self).__init__()
        self.model_name = 'basenet'
        if depth == 18:
            self.backbone = resnet18(pretrained)
        elif depth == 50:
            self.backbone = resnet50(pretrained)
        self.reduce_s1 = Reduction(64, channel)
        self.reduce_s2 = Reduction(128, channel)
        self.reduce_s3 = Reduction(256, channel)
        self.reduce_s4 = Reduction(512, channel)

        self.reduce_e1 = Reduction(64, channel)
        self.reduce_e2 = Reduction(128, channel)
        self.reduce_e3 = Reduction(256, channel)
        self.reduce_e4 = Reduction(512, channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # here will add DF module
        # x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e


class BaseNetWithCRU(BasicModule):
    def __init__(self, depth=18, pretrained=True, channel=32):
        super(BaseNetWithCRU, self).__init__()
        self.model_name = 'basenet'
        if depth == 18:
            self.backbone = resnet18(pretrained)
        elif depth == 50:
            self.backbone = resnet50(pretrained)
        self.reduce_s1 = Reduction(64, channel)
        self.reduce_s2 = Reduction(128, channel)
        self.reduce_s3 = Reduction(256, channel)
        self.reduce_s4 = Reduction(512, channel)

        self.reduce_e1 = Reduction(64, channel)
        self.reduce_e2 = Reduction(128, channel)
        self.reduce_e3 = Reduction(256, channel)
        self.reduce_e4 = Reduction(512, channel)

        self.df1 = DenseFusion(channel)
        self.df2 = DenseFusion(channel)
        self.df3 = DenseFusion(channel)
        self.df4 = DenseFusion(channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # here will add DF module
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, pred_e


class BaseNetWithBRU(BasicModule):
    def __init__(self, depth=18, pretrained=True, channel=32, deepsup=False):
        super(BaseNetWithBRU, self).__init__()
        self.model_name = 'basenet'
        if depth == 18:
            self.backbone = resnet18(pretrained)
        elif depth == 50:
            self.backbone = resnet50(pretrained)
        self.deepsup = deepsup
        self.reduce_s1 = Reduction(64, channel)
        self.reduce_s2 = Reduction(128, channel)
        self.reduce_s3 = Reduction(256, channel)
        self.reduce_s4 = Reduction(512, channel)

        self.reduce_e1 = Reduction(64, channel)
        self.reduce_e2 = Reduction(128, channel)
        self.reduce_e3 = Reduction(256, channel)
        self.reduce_e4 = Reduction(512, channel)

        self.df1 = DF(channel)
        self.df2 = DF(channel)
        self.df3 = DF(channel)
        self.df4 = DF(channel)

        if self.deepsup:
            self.output1_s = ConcatOutput(channel)
            self.output1_e = ConcatOutput(channel)
            self.output2_s = ConcatOutput(channel)
            self.output2_e = ConcatOutput(channel)
            self.output3_s = ConcatOutput(channel)
            self.output3_e = ConcatOutput(channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # here will add DF module
        preds_s, preds_e = [], []
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        if self.deepsup:
            pred1_s, pred1_e = self.output1_s(x_s1, x_s2, x_s3, x_s4), self.output1_e(x_e1, x_e2, x_e3, x_e4)
            pred1_s, pred1_e = F.upsample(pred1_s, size=size, mode='bilinear', align_corners=True), \
                               F.upsample(pred1_e, size=size, mode='bilinear', align_corners=True)
            preds_s.append(pred1_s), preds_e.append(pred1_e)

        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        if self.deepsup:
            pred2_s, pred2_e = self.output2_s(x_s1, x_s2, x_s3, x_s4), self.output2_e(x_e1, x_e2, x_e3, x_e4)
            pred2_s, pred2_e = F.upsample(pred2_s, size=size, mode='bilinear', align_corners=True), \
                               F.upsample(pred2_e, size=size, mode='bilinear', align_corners=True)
            preds_s.append(pred2_s), preds_e.append(pred2_e)

        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        if self.deepsup:
            pred3_s, pred3_e = self.output3_s(x_s1, x_s2, x_s3, x_s4), self.output3_e(x_e1, x_e2, x_e3, x_e4)
            pred3_s, pred3_e = F.upsample(pred3_s, size=size, mode='bilinear', align_corners=True), \
                               F.upsample(pred3_e, size=size, mode='bilinear', align_corners=True)
            preds_s.append(pred3_s), preds_e.append(pred3_e)

        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        if self.deepsup:
            return pred_s, pred_e, preds_s, preds_e
        else:
            return pred_s, pred_e


def test():
    net = BaseNet(True, (224, 224))
    x = torch.randn(1, 3, 224, 224)
    pred_s, pred_e = net(x)
    net.save('checkpoints/basenet')
    # print(c.shape)

    # print(net) #

    # for name, param in net.named_parameters():
    #     print(name)

    # for name, module in net.named_children():
    #     print(name)
    # backbone
    # neck
    # head

    # num_params = 0
    # for p in net.parameters():
    #     num_params += p.numel()
    # print("The number of parameters: {}".format(num_params))


if __name__ == '__main__':
    test()
