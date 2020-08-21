import torch
import torch.nn as nn
import time
import os


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, root=None):
        if root is None:
            root = 'checkpoints/' + self.model_name
        name = time.strftime('%m%d_%H:%M:%S.pth')
        if not os.path.exists(root):
            print('Makedirs ', root)
            os.makedirs(root)
        path = os.path.join(root, name)
        torch.save(self.state_dict(), path)
        print('Model has been saved in ', path)
        return path
