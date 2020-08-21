import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

__all__ = ['EdgeLoss']


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, output, target, t=0.5):
        # output [batch_size,1,h,w], target:edge mask 0,1  tensor
        output = torch.sigmoid(output)  # 0-1,torch.float32
        neg_target = ~ target.type(torch.ByteTensor)

        target = target.data.cpu().numpy()
        output = output.data.cpu().numpy()
        # target = target.data.cpu().numpy()
        neg_target = neg_target.data.cpu().numpy()

        # calculate e distmap
        # in_dist = distance(target).astype(np.float32)  # ndarray
        # out_dist = distance(neg_target).astype(np.float32)  # ndarray
        # dist = in_dist + out_dist

        # dist = (distance(neg_target) - ((distance(target) - 1) * target)).astype(np.float32)
        dist = (distance(neg_target) - (distance(target))).astype(np.float32)

        # dist = torch.from_numpy(dist)
        # output = torch.from_numpy(output)
        # dist.type_as(output)

        mul = np.einsum('bchw,bchw->bchw', output, dist)
        loss = mul.mean()

        # 加shang

        return loss


def test():
    target = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1., 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]]]).cuda()
    print(target.shape)
    neg_target = ~ target.type(torch.ByteTensor)
    print(neg_target)

    output = torch.tensor([[[[0., 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]]]).cuda()

    output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()
    neg_target = neg_target.data.cpu().numpy()

    dist = (distance(neg_target) - ((distance(target) - 1) * target)).astype(np.float32)
    print(dist)
    mul = np.einsum('bchw,bchw->bchw', output, dist)
    print(mul.mean())


if __name__ == '__main__':
    test()
