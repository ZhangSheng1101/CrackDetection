import numpy as np
import torch


def iou_score(output, target, epsilon=1e-5, t=0.5):
    # shape = [batch_size,1,h,w]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    assert output.shape == target.shape
    output_ = output > t
    target_ = target > t

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    # print(output.shape[0])
    # print((intersection + epsilon) / (union + epsilon))
    #
    # batch_size = output.shape[0]
    #
    # output_ = output_.reshape(batch_size, -1)
    # target_ = target_.reshape(batch_size, -1)
    # intersection = (output_ & target_).sum(1)
    # union = (output_ | target_).sum(1)

    return (intersection + epsilon) / (union + epsilon)
    # print(((intersection + epsilon) / (union + epsilon)).sum() / batch_size)


def test():
    a = np.random.randn(5, 1, 640, 360)
    b = np.random.randn(5, 1, 640, 360)
    print(iou_score(a, b))


if __name__ == '__main__':
    test()
