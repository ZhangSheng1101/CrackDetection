import torch
import torch.nn as nn


def f_score(output, target, beta=0.3):
    batch_size = output.size(0)
    output = output.view(batch_size, -1)
    target = target.view(batch_size, -1)

    T = output.sum(1) * (2 / (output.size(1)))
    output = output > T

    intersection = (output * target).sum(1)
    precision = intersection / output.sum(1)
    recall = intersection / target.sum(1)

    f = (((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)).sum() / batch_size

    return f


if __name__ == '__main__':
    a = torch.randn(5, 1, 4, 4)
    a = a.view(5, -1)
    a = a.sum(0)
    print(a.shape)
