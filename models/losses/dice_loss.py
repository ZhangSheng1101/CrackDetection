import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        assert output.size() == target.size()
        batch_size = output.size(0)
        output = torch.sigmoid(output)  # 0-1

        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        intersection = output * target
        # print(batch_size, intersection.sum(1), output.sum(1), target.sum(1))
        dice = (2. * intersection.sum(1) + self.smooth) / (output.sum(1) + target.sum(1) + self.smooth)
        dice = 1 - dice.sum() / batch_size

        return dice


def test():
    a = torch.ones(5, 1, 20, 20)
    b = torch.ones(5, 1, 20, 20)
    loss = DiceLoss()
    print(loss(a, b))


if __name__ == '__main__':
    test()
