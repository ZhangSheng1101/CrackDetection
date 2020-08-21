# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import utils as vutils
# project
from config import config
from datasets import CRACK500
from models import *
from tensorboardX import SummaryWriter
from evaluation import iou_score
from utils import AverageMeter, ProgressMeter, mask2edge
from models import DiceLoss, EdgeLoss
#
import os
from tqdm import tqdm
import time
import shutil
import numpy as np
import random
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # seed
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
    # model = BaseNet(config.pretained, config.img_size)
    model = BaseNetWithBRU(deepsup=True)

    # 暂时多GPU训练先用这个
    if config.data_parallel:
        print('Now use', torch.cuda.device_count(), 'GPUs!')
        print('Each GPU will process', config.batch_size // torch.cuda.device_count(), 'images.')
        model = nn.DataParallel(model)

    if config.use_gpu:
        model.cuda()

    # learningrate settings
    # backbone, head = [], []
    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         backbone.append(param)
    #     else:
    #         head.append(param)
    #
    # # optimizer
    # optimizer = torch.optim.SGD([
    #     {'params': backbone, 'lr': config.backbone_lr},
    #     {'params': head}
    # ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # 考虑到和原imagenet图像数据集差别较大，backbone学习率和head调整至一致
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step_size,
                                                gamma=config.lr_decay_gamma)

    # define loss function
    if config.loss == 'BCEWithLogitsLoss':
        print('Salient loss use BCE, Edge loss use BCE.')
        criterion_s = nn.BCEWithLogitsLoss()
        criterion_e = nn.BCEWithLogitsLoss()
    elif config.loss == 'DiceLoss':
        print('Salient loss use Dice, Edge loss use BCE')
        criterion_e = nn.BCEWithLogitsLoss()
        criterion_s = DiceLoss()
    elif config.loss == 'DiceBoundaryLoss':
        print('Salient loss use Dice, Edge loss use BoundaryLoss')
        criterion_e = EdgeLoss()
        criterion_s = DiceLoss()

    if config.use_gpu:
        criterion_s.cuda()
        criterion_e.cuda()

    # data
    train_data = CRACK500(config.data_root, config.img_size, mode='train', seed=config.seed)
    print('Train dataset contains %d images' % len(train_data))
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers,
                              pin_memory=True)

    val_data = CRACK500(config.data_root, config.img_size, mode='test', seed=config.seed)  # an shi buyong val le
    print('Validate dataset contains %d images' % len(val_data))
    val_loader = DataLoader(val_data, config.batch_size, shuffle=False, num_workers=config.num_workers,
                            pin_memory=True)

    # tensorboard
    train_log_dir = os.path.join(config.tensorboard_outdir, config.model_name, 'train')
    val_log_dir = os.path.join(config.tensorboard_outdir, config.model_name, 'val')
    if os.path.exists(train_log_dir):
        shutil.rmtree(train_log_dir)
    os.makedirs(train_log_dir)
    if os.path.exists(val_log_dir):
        shutil.rmtree(val_log_dir)
    os.makedirs(val_log_dir)

    trainwriter = SummaryWriter(log_dir=train_log_dir)
    valwriter = SummaryWriter(log_dir=val_log_dir)

    best_iou = 0.
    for epoch in range(config.epochs):
        train(train_loader, model, criterion_s, criterion_e, optimizer, epoch, trainwriter)
        scheduler.step()
        # validate
        val_iou = validate(val_loader, model, criterion_s, criterion_e, epoch, valwriter)
        if val_iou > best_iou:
            best_iou = val_iou
            print('val_iou', val_iou, 'is best now! Now save the model')
            if config.data_parallel:
                # 只保存module里的
                save_checkpoints(model.module.state_dict(), config.save_model_path)
                # model.module.save(os.path.join(config.save_model_root, config.model_name))
            else:
                save_checkpoints(model.state_dict(), config.save_model_path)
        torch.cuda.empty_cache()
        print('-' * 20)

    print('TRAIN DONE.')


def train(train_loader, model, criterion_s, criterion_e, optimizer, epoch, trainwriter):
    losses = AverageMeter('Loss', ':.6f')
    losses_e = AverageMeter('Loss_edge', ':.6f')
    losses_s = AverageMeter('Loss_salient', ':.6f')
    ious = AverageMeter('IoU', ':.4f')
    pregress = ProgressMeter(
        len(train_loader), [losses, losses_e, losses_s, ious],
        prefix='Epoch: [{}/{}]'.format(epoch, config.epochs))

    model.train()
    time_start = time.time()

    for step, (image, mask, _, _) in enumerate(tqdm(train_loader, ncols=80, ascii=True)):
        # for i in range(3):
        #     vutils.save_image(image[i], 'i' + str(i) + '.png')
        #     vutils.save_image(mask[i], 'm' + str(i) + '.png')
        #     vutils.save_image(edge[i], 'e' + str(i) + '.png')
        if config.use_gpu:
            image, mask = image.cuda(), mask.cuda()

        edge = mask2edge(mask)
        if not config.deepsupvision:
            pred_s, pred_e = model(image)  # [batch_size, 1, h, w]
            # pred_s = model(image)
            # loss
            # loss_s = criterion(pred_s, mask)
            loss_s = criterion_s(pred_s, mask)
            loss_e = criterion_e(pred_e, edge)
        # if epoch < 20:
        #     weight = 1. - epoch * 0.01
        #     loss = loss_s * weight + (1 - weight) * loss_e / 4
        # else:
        #     weight = 1. - (epoch - 20) * 0.09
        #     loss = weight * loss_s + (1 - weight) * loss_e / 4
        else:
            pred_s, pred_e, preds_s, preds_e = model(image)
            loss_s = criterion_s(pred_s, mask)
            loss_e = criterion_e(pred_e, edge)
            for i in range(len(preds_s)):
                loss_s += 0.3 * criterion_s(preds_s[i], mask)
                loss_e += 0.3 * criterion_e(preds_e[i], edge)
        weight = epoch * 0.03
        loss = loss_s + loss_e * weight
        # iou
        iou = iou_score(pred_s, mask)

        #################################

        # update
        batch_size = image.size(0)
        losses_e.update(loss_e.item(), batch_size)
        losses_s.update(loss_s.item(), batch_size)
        losses.update(loss.item(), batch_size)
        ious.update(iou, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if step % config.print_freq == 0:
            pregress.display(step)
            vutils.save_image(pred_s[0], 'temp/iter_' + str(step) + 'step.png')
            vutils.save_image(pred_e[0], 'temp/iter_' + str(step) + 'step_edge.png')

    time_end = time.time()

    trainwriter.add_scalar('epoch loss', losses.avg, epoch + 1)
    trainwriter.add_scalar('epoch loss_s', losses_s.avg, epoch + 1)
    trainwriter.add_scalar('epoch loss_e', losses_e.avg, epoch + 1)
    trainwriter.add_scalar('iou', ious.avg, epoch + 1)

    # print('Epoch %d consume %d seconds. lr = %.8f, bk_lr = %.8f' % (
    #     epoch + 1, time_end - time_start, optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr']))
    print('Epoch %d consume %d seconds. lr = %.6f, losses = %.6f, losses_e = %.6f, losses_s = %.6f, iou = %.2f%%.' % (
        epoch + 1, time_end - time_start, optimizer.param_groups[0]['lr'], losses.avg, losses_e.avg, losses_s.avg,
        ious.avg * 100))


def validate(val_loader, model, criterion_s, criterion_e, epoch, valwriter):
    losses = AverageMeter('Loss', ':.6f')
    losses_e = AverageMeter('Loss_edge', ':.6f')
    losses_s = AverageMeter('Loss_salient', ':.6f')
    ious = AverageMeter('IoU', ':.4f')
    # pregress = ProgressMeter(
    #     len(val_loader), [losses, losses_e, losses_s, ious],
    #     prefix='Epoch: [{}/{}]'.format(epoch + 1, config.epochs))
    #

    model.eval()
    with torch.no_grad():
        for step, (image, mask, _, _) in enumerate(tqdm(val_loader, ncols=80, ascii=True)):
            if config.use_gpu:
                image, mask = image.cuda(), mask.cuda()
            edge = mask2edge(mask)
            if not config.deepsupvision:
                pred_s, pred_e = model(image)  # [batch_size, 1, h, w]
                # pred_s = model(image)
                # loss
                # loss_s = criterion(pred_s, mask)
                loss_s = criterion_s(pred_s, mask)
                loss_e = criterion_e(pred_e, edge)
            # if epoch < 20:
            #     weight = 1. - epoch * 0.01
            #     loss = loss_s * weight + (1 - weight) * loss_e / 4
            # else:
            #     weight = 1. - (epoch - 20) * 0.09
            #     loss = weight * loss_s + (1 - weight) * loss_e / 4
            else:
                pred_s, pred_e, preds_s, preds_e = model(image)
                loss_s = criterion_s(pred_s, mask)
                loss_e = criterion_e(pred_e, edge)
                for i in range(len(preds_s)):
                    loss_s += 0.3 * criterion_s(preds_s[i], mask)
                    loss_e += 0.3 * criterion_e(preds_e[i], edge)
            weight = epoch * 0.03
            loss = loss_s + loss_e * weight
            #
            iou = iou_score(pred_s, mask)

            # update
            batch_size = image.size(0)
            losses_e.update(loss_e.item(), batch_size)
            losses_s.update(loss_s.item(), batch_size)
            losses.update(loss.item(), batch_size)
            ious.update(iou, batch_size)

    valwriter.add_scalar('epoch loss', losses.avg, epoch + 1)
    valwriter.add_scalar('epoch loss_s', losses_s.avg, epoch + 1)
    valwriter.add_scalar('epoch loss_e', losses_e.avg, epoch + 1)
    valwriter.add_scalar('iou', ious.avg, epoch + 1)
    print('VAL  Epoch:[{:d}/{:d}], loss:{:.6f}, loss_e:{:.6f}, loss_s:{:.6f}, iou:{:.2f}%'
          .format(epoch + 1, config.epochs, losses.avg, losses_e.avg, losses_s.avg, ious.avg * 100))
    return ious.avg


def save_checkpoints(state, fileName):
    torch.save(state, fileName)


if __name__ == '__main__':
    main()
