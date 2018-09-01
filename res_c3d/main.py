import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import numpy as np
from torch.autograd import Variable

from opt import parse_opts
from resnet18_3d import ResNet
from dataset import demo_dataset
from valida_trained_model import get_demo_dataloader
from utils import calculate_accuracy
# from spatial_transform import Normalize, Compose, MultiScaleCornerCrop, MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, Scale, CenterCrop
# from temporal_transform import TemporalRandomCrop, LoopPadding
# from target_transform import ClassLabel
# from utils import AverageMeter, Logger
# from dataset import UCF101
from train import get_trainingset, train_network
from validation import get_validationset, validate_network
from k_folder_cross_validation import k_folder_cross_validation
# import k_folder_cross_validation


def get_mean_std(opt):
    assert opt.mean_dataset  in ['activitynet', 'kinetics']
    assert opt.norm_value == 1 or opt.norm_value == 255
    mean = {'activitynet':[114.7748 / opt.norm_value, 107.7354 / opt.norm_value, 99.4750 / opt.norm_value], 'kinetics':[110.63666788 / opt.norm_value, 103.16065604 / opt.norm_value, 96.29023126 / opt.norm_value]}
    std = [38.7568578 / opt.norm_value, 37.88248729 / opt.norm_value, 40.02898126 / opt.norm_value]

    if opt.no_mean_norm:
        mean[opt.mean_dataset] = [0, 0, 0]
    if opt.std_norm:
        std = [1, 1, 1]

    return mean[opt.mean_dataset], std


def generate_model(opt):
    print('generating model, model type is: {}, model depth is: {}'.format(opt.model, opt.model_depth))
    model = ResNet(opt, sample_duration=opt.sample_duration,
                   sample_size=opt.sample_size)

    # use cuda
    assert torch.cuda.is_available() == True
    model = model.cuda(opt.device[0])
    model = nn.DataParallel(model, device_ids=opt.device)
    #
    model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
    model.module.fc = model.module.fc.cuda(opt.device[0])


    # use pretrained weights
    assert os.path.exists(opt.pretrain_path)
    # print('loading pretrained model from: {}'.format(opt.pretrain_path))
    pretrain_model = torch.load(opt.pretrain_path, map_location=lambda storage, loc: storage.cuda(opt.device[0]))

    assert opt.arch == pretrain_model['arch']
    model.load_state_dict(pretrain_model['state_dict'])

    # model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
    # model.module.fc = model.module.fc.cuda(opt.device[0])

    # freeze part of the model
    parameters = []
    assert opt.ft_begin_index <= 5
    if opt.ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(opt.ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return model, parameters

#
# def get_trainingset(opt):
#     print('training~~~')
#     assert opt.train_crop in ['random', 'corner', 'center']
#     norm_method = Normalize(opt.mean, opt.std)
#     if opt.train_crop == 'random':
#         crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
#     elif opt.train_crop == 'corner':
#         crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
#     elif opt.train_crop == 'center':
#         crop_method = MultiScaleCornerCrop(
#             opt.scales, opt.sample_size, crop_positions=['c'])
#     spatial_transform = Compose([
#         crop_method,
#         RandomHorizontalFlip(),
#         ToTensor(opt.norm_value), norm_method
#     ])
#     temporal_transform = TemporalRandomCrop(opt.sample_duration)
#     target_transform = ClassLabel()
#
#     training_data = UCF101(
#         opt.video_path,
#         opt.annotation_path,
#         'training',
#         spatial_transform,
#         temporal_transform,
#         target_transform)
#
#     train_loader = torch.utils.data.DataLoader(
#         training_data,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.n_threads,
#         pin_memory=True)
#
#     train_logger = Logger(
#         os.path.join(opt.result_path, 'train.log'),
#         ['epoch', 'loss', 'acc', 'lr'])
#     train_batch_logger = Logger(
#         os.path.join(opt.result_path, 'train_batch.log'),
#         ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
#
#     # return train_loader, train_logger, train_batch_logger


# def get_validationset(opt):
#     print('validating~~~~')
#     norm_method = Normalize(opt.mean, opt.std)
#     spatial_transform = Compose([
#         Scale(opt.sample_size),
#         CenterCrop(opt.sample_size),
#         ToTensor(opt.norm_value), norm_method
#     ])
#     temporal_transform = LoopPadding(opt.sample_duration)
#     target_transform = ClassLabel()
#     validation_data = UCF101(
#         root_path=opt.video_path,
#         annotation_path=opt.annotation_path,
#         subset='validation',
#         spatial_transform=spatial_transform,
#         temporal_transform=temporal_transform,
#         target_transform=target_transform,
#         n_samples_for_each_video=opt.n_val_samples,
#         sample_duration=opt.sample_duration
#     )
#     val_loader = torch.utils.data.DataLoader(
#         validation_data,
#         batch_size=opt.batch_size,
#         shuffle=False,
#         num_workers=opt.n_threads,
#         pin_memory=True)
#     val_logger = Logger(
#         os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
#     return val_loader, val_logger

#
# def calculate_accuracy(target, predict, top_n=1):
#     assert len(target) == len(predict)
#     n_samples = len(target)


#
# def train_network(opt, optimizer, dataloader, criertion, epoch_logger ,epoch=1):
#     for i, item in enumerate(dataloader):
#         start_time = time.time()
#
#         input =


if __name__ == '__main__':

    # init opt
    opt = parse_opts()
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean, opt.std = get_mean_std(opt)
    # print(opt)

    torch.manual_seed(opt.manual_seed)

    # init model and criterion
    # here the learning rate is 0 or no such a dict 'lr' is really common
    # since after the optimizer the lr or momentum etc will be added
    model, parameters = generate_model(opt)
    criterion = nn.CrossEntropyLoss().cuda(opt.device[0])

    # opt.no_train, opt.no_val = True, True
    # opt.no_val = True

    # init for training
    if not opt.no_train:
        # train_loader, train_logger, train_batch_logger = get_trainingset(opt)
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

    # init for validation
    if not opt.no_val and opt.dataset != 'pingpong_dataset':
        pass
        # val_loader, val_logger = get_validationset(opt)

    # since I adopt cross validation, so the process to get training set and validation set is different
    if opt.dataset == 'pingpong_dataset':
        cross_validation = k_folder_cross_validation(5, opt, dataset='pingpong_dataset')
        train_loader, train_logger, train_batch_logger, val_loader, val_logger = cross_validation()

    demo_loader = get_demo_dataloader(opt)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            print('training at epoch: {}'.format(i))

            train_network(opt=opt,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          train_logger=train_logger,
                          dataloader=train_loader,
                          epoch=i)

        if not opt.no_val:
            print('validation at epoch: {}'.format(i))
            val_loss = validate_network(opt=opt,
                             model=model,
                             dataloader=val_loader,
                             val_log=val_logger,
                             criterion=criterion,
                             epoch=i)

        if i % 5 == 1:
            demo_data = []
            for index, (input, target, meta_data) in enumerate(demo_loader):

                model.eval()
                input = Variable(input.cuda(opt.device[0]), volatile=True)
                target = Variable(target.cuda(opt.device[0]), volatile=True)

                output = model(input)

                # loss = criterion(output, target)
                acc = calculate_accuracy(output, target)

                for i in range(len(meta_data)):
                    info = meta_data[i].split(';')
                    demo_data.append({'info':
                                          {'path':info[0],
                                           'start_time': int(info[1]),
                                           'end_time':int(info[2]),
                                           'ground_truth':int(info[3])},
                                      'predict':[float(item) for item in output.data.cpu()[i]],
                                      'ground_truth': int(target.data.cpu()[i])})
                # losses.update(loss.data[0])
                print(index, acc)

            with open('./result.json', mode='w') as f:
                json.dump(demo_data, f)


        if not opt.no_train and not opt.no_val:
            print('scheduler step')
            scheduler.step(val_loss)


    # for get the features extracted from the network
    # if opt.no_val and opt.no_train:
    #
    #     train_data, train_label = None, None
    #     test_data, test_label = None, None
    #
    #     # val_loader, val_logger = get_validationset(opt)
    #     #
    #     # for index, (input, result) in enumerate(val_loader):
    #     #     print('[{}]/[{}]'.format(index, len(val_loader)))
    #     #
    #     #     input = Variable(input)
    #     #
    #     #     output = model(input)
    #     #     output = torch.Tensor.numpy(output.cpu().data)
    #     #
    #     #     result = torch.LongTensor.numpy(result)
    #     #
    #     #     try:
    #     #         test_data = np.concatenate((test_data, output), axis=0)
    #     #         test_label = np.concatenate((test_label, result), axis=0)
    #     #     except:
    #     #         test_data = output
    #     #         test_label = result
    #     #
    #     # np.savetxt(fname='test_data.txt', X=test_data)
    #     # np.savetxt(fname='test_label.txt', X=test_label)
    #
    #     # del test_data
    #     # del test_label
    #
    #     train_loader, train_logger, train_batch_logger = get_trainingset(opt)
    #
    #     for index, (input, result) in enumerate(train_loader):
    #         print('[{}]/[{}]'.format(index, len(train_loader)))
    #
    #         input = Variable(input)
    #
    #         output = model(input)
    #         output = torch.Tensor.numpy(output.cpu().data)
    #
    #         result = torch.LongTensor.numpy(result)
    #
    #         try:
    #             train_data = np.concatenate((train_data, output), axis=0)
    #             train_label = np.concatenate((train_label, result), axis=0)
    #         except:
    #             train_data = output
    #             train_label = result
    #
    #     np.savetxt(fname='train_data.txt',X=train_data)
    #     np.savetxt(fname='train_label.txt', X=train_label)
