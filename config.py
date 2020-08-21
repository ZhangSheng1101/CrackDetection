import warnings


class DefaultConfig(object):
    # model
    model_name = 'base'
    pretained = True

    # data
    data_root = './data/CRACK500'
    img_size = (352, 640)  # (h,w)

    #
    batch_size = 16
    use_gpu = True
    data_parallel = False
    num_workers = 4
    print_freq = 10

    # train
    seed = 1
    epochs = 30
    backbone_lr = 2e-5  # for vgg, 5e-5 for resnet
    lr = 2e-3  #
    weight_decay = 5e-4
    momentum = 0.9
    # lr_scheduler
    lr_decay_step_size = 20
    lr_decay_gamma = 0.1
    loss = 'DiceBoundaryLoss'
    save_model_path = 'checkpoints/basenet_dice_bl_with_brus.pth'


    deepsupvision = True

    # visualization
    tensorboard_outdir = 'logs'
    tf_step_size = 20

    # test
    load_model_path = 'checkpoints/basenet_dice_bl_with_brus.pth'
    save_images = True
    save_images_root = 'outs'


def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
config = DefaultConfig()
