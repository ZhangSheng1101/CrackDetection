# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision.utils as vutils
# project
from config import config
from datasets import CRACK500
from models import *
from tensorboardX import SummaryWriter
from evaluation import iou_score
from utils import AverageMeter, ProgressMeter
#
import os
from tqdm import tqdm
import time
import shutil
from PIL import Image
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test():
    random.seed(config.seed)
    np.random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # model
    print('-' * 10, 'create model', '-' * 10)
    # model = BaseNet(False, config.img_size)
    model = BaseNetWithBRU(deepsup=config.deepsupvision)

    # 暂时多GPU训练先用这个
    if config.data_parallel:
        print('Now use', torch.cuda.device_count(), 'GPUs!')
        print('Each GPU will process', config.batch_size // torch.cuda.device_count(), 'images.')
        model = nn.DataParallel(model)

    if config.use_gpu:
        model.cuda()

    # load trained model
    print('Load trained model from', config.load_model_path)
    if config.data_parallel:
        model.module.load_state_dict(torch.load(config.load_model_path))
    else:
        model.load_state_dict(torch.load(config.load_model_path))

    # data
    test_data = CRACK500(config.data_root, config.img_size, mode='test', seed=config.seed)
    print('Test dataset contains %d images' % len(test_data))
    test_loader = DataLoader(test_data, config.batch_size, shuffle=False, num_workers=config.num_workers,
                             pin_memory=True)

    # tensorboard maybe not use

    ious = AverageMeter('IoU', ':.4f')

    model.eval()
    with torch.no_grad():
        for step, (image, mask, _, name) in enumerate(tqdm(test_loader, ncols=80, ascii=True)):
            if config.use_gpu:
                image, mask = image.cuda(), mask.cuda()
            if not config.deepsupvision:
                pred_s, pred_e = model(image)
            else:
                pred_s, pred_e, _, _ = model(image)
            # pred_s = model(image)

            iou = iou_score(pred_s, mask)

            batch_size = image.size(0)
            ious.update(iou, batch_size)

            if config.save_images:
                save_path = os.path.join(config.save_images_root, model.model_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                batch_size = image.size(0)
                pred_s = torch.sigmoid(pred_s)
                pred_e = torch.sigmoid(pred_e)
                for i in range(batch_size):
                    vutils.save_image(pred_s[i], os.path.join(save_path, name[i]))
                    vutils.save_image(pred_e[i], os.path.join(save_path, name[i][:-4] + '_edge' + '.png'))

    print('Test Iou:{:.2f}%'.format(ious.avg * 100))


def test_one_img():
    print('-' * 10, 'create model', '-' * 10)
    # model = BaseNet(False, config.img_size)
    model = BaseNetWithBRU()

    # 暂时多GPU训练先用这个
    if config.data_parallel:
        print('Now use', torch.cuda.device_count(), 'GPUs!')
        print('Each GPU will process', config.batch_size // torch.cuda.device_count(), 'images.')
        model = nn.DataParallel(model)

    if config.use_gpu:
        model.cuda()

    # load trained model
    print('Load trained model from', config.load_model_path)
    if config.data_parallel:
        model.module.load_state_dict(torch.load(config.load_model_path))
    else:
        model.load_state_dict(torch.load(config.load_model_path))

    # data
    test_data = CRACK500(config.data_root, config.img_size, mode='test', seed=config.seed)
    print('Test dataset contains %d images' % len(test_data))
    test_loader = DataLoader(test_data, 1, shuffle=False, num_workers=config.num_workers,
                             pin_memory=True)

    # tensorboard maybe not use

    ious = AverageMeter('IoU', ':.4f')

    model.eval()
    with torch.no_grad():
        for step, (image, mask, edge) in enumerate(tqdm(test_loader, ncols=80, ascii=True)):
            if config.use_gpu:
                image, mask, edge = image.cuda(), mask.cuda(), edge.cuda()
            pred_s, pred_e = model(image)
            # pred_s = model(image)

            iou = iou_score(pred_s, mask)

            batch_size = image.size(0)
            ious.update(iou, batch_size)

            pred_s = torch.sigmoid(pred_s)
            pred_e = torch.sigmoid(pred_e)
            for i in range(len(pred_s)):
                ps = pred_s[i, 0].data.cpu().numpy()
                pe = pred_e[i, 0].data.cpu().numpy()
                ps = ps > 0.5
                pe = pe > 0.5
                s = Image.fromarray(ps).convert('L')
                e = Image.fromarray(pe).convert('L')

                s.save('se.png')
                e.save('ee.png')

            return


if __name__ == '__main__':
    test()
