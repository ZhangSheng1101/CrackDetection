import torch
import torch.nn.functional as F
import numpy as np


def mask2edge(mask, t=0.1):
    # [batch_size,1,h,w], tensor.float32 0,1
    mask = mask.gt(0.5).float()
    mask = F.pad(mask, (1, 1, 1, 1), mode='replicate')

    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = torch.from_numpy(fx).cuda()
    fy = torch.from_numpy(fy).cuda()

    mask_fx = F.conv2d(mask, fx)
    mask_fy = F.conv2d(mask, fy)

    mask_grad = torch.sqrt(torch.mul(mask_fx, mask_fx) + torch.mul(mask_fy, mask_fy))
    edge = torch.gt(mask_grad, t).float()

    return edge
