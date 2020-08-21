import os
import torch.utils.data as data
from PIL import Image
from . import my_transforms as mt


class CRACK500(data.Dataset):
    def __init__(self, root, img_size, mode='train', transforms=None, seed=1):
        self.root = root
        self.images_dir = os.path.join(self.root, mode, 'images')
        self.targets_dir = os.path.join(self.root, mode, 'masks')
        if transforms is None:
            if mode == 'train':
                transforms = mt.Compose([
                    mt.Resize(img_size),
                    mt.RandomHorizontalFlip(seed),
                    mt.ToTensor(),  # from (H, W, C) 0-255 to a torch.FloatTensor (C,H,W) 0-1
                    # normlize
                    mt.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
            elif mode == 'val':
                transforms = mt.Compose([
                    mt.Resize(img_size),
                    mt.ToTensor(),  # from (H, W, C) 0-255 to a torch.FloatTensor (C,H,W) 0-1
                    # normlize
                    mt.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
            else:
                transforms = mt.Compose([
                    mt.Resize(img_size),
                    mt.ToTensor(),
                    # normlize
                    mt.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
        self.transforms = transforms  #

        self.images = []
        self.targets = []
        self.edges = []
        self.names = []

        for img_name in os.listdir(self.images_dir):
            img_id = img_name[:-4]
            target_name = img_id + '.png'
            edge_name = img_id + '_edge' + '.png'
            self.images.append(os.path.join(self.images_dir, img_name))
            self.targets.append(os.path.join(self.targets_dir, target_name))
            self.edges.append(os.path.join(self.targets_dir, edge_name))
            self.names.append(target_name)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # PIL Image object
        target = Image.open(self.targets[index]).convert('L')
        edge = Image.open(self.edges[index]).convert('L')

        if self.transforms is not None:
            image, target, edge = self.transforms(image, target, edge)
        # return image, target, edge
        return image, target, edge, self.names[index]

    def __len__(self):
        return len(self.images)


# class StandardTransform(object):
#     def __init__(self, transform, target_transform):
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __call__(self, image, target, edge):
#         if self.transform is not None and self.target_transform is not None:
#             image = self.transform(image)
#             target = self.target_transform(target)
#             edge = self.target_transform(edge)
#
#         return image, target, edge


def test():
    data_ = CRACK500('testcrack500', (360, 640))
    # print(e)
    train_loader = data.DataLoader(data_, batch_size=5)
    for idx, (i, m, e, _) in enumerate(train_loader):
        print(i.shape, m.shape, e.shape)
    # img = Image.open('testcrack500/train/images/20160222_081011_1_361.jpg')
    # mask = Image.open('testcrack500/train/images/20160222_081011_1_361.jpg')
    # edge = Image.open('testcrack500/train/images/20160222_081011_1_361.jpg')
    # transforms = mt.Compose([
    #     mt.RandomHorizontalFlip(),
    #     mt.ToTensor(),  # from (H, W, C) 0-255 to a torch.FloatTensor (C,H,W) 0-1
    #     # mt.Normalize(mean=[0.485, 0.456, 0.406],
    #     #              std=[0.229, 0.224, 0.225])
    # ])
    # i, m, e = transforms(img, mask, edge)
    # print((np.array(img) == np.array(mask)).all())
    # print((np.array(edge) == np.array(mask)).all())
    # print((i == m).all())
    # print((i == e).all())
    # print((m == e).all())
    # print(m)
    return


if __name__ == '__main__':
    test()
