import torch.nn as nn
import torch


class TestClass(nn.Module):
    def __init__(self):
        super(TestClass, self).__init__()
        conv = []
        conv.append(None)
        conv.append(nn.Conv2d(3, 3, 3, 1))

        self.conv = nn.ModuleList(conv)
        print(self.conv)
    def forward(self, x):
        return self.conv[-1](x)


def test():
    x = torch.randn(1, 3, 224, 224)
    net = TestClass()
    out = net(x)

    print(out.shape)


if __name__ == '__main__':
    test()
