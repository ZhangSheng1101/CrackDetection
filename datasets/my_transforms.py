import torch
import torchvision.transforms.functional as TF
import random
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        for t in self.transforms:
            img, mask, edge = t(img, mask, edge)
        return img, mask, edge


class ToTensor(object):
    def __call__(self, img, mask, edge):
        img, mask, edge = TF.to_tensor(img), TF.to_tensor(mask), TF.to_tensor(edge)
        return img, mask, edge


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img_tensor, mask_tensor, edge_tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # mask和edge不需要经过normalize
        return TF.normalize(img_tensor, self.mean, self.std, self.inplace), mask_tensor, edge_tensor


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        # assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, mask, edge):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return TF.resize(img, self.size, self.interpolation), TF.resize(mask, self.size, self.interpolation), TF.resize(
            edge, self.size, self.interpolation)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, seed=1, p=0.5):
        self.p = p
        random.seed(seed)

    def __call__(self, img, mask, edge):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask), TF.hflip(edge)
        return img, mask, edge
