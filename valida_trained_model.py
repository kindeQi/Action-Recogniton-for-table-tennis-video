from opt import parse_opts
from resnet18_3d import ResNet
from k_folder_cross_validation import k_folder_cross_validation

# from main import get_mean_std

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
from dataset import UCF101, pingpong_dataset, demo_dataset
from utils import Logger
from spatial_transform import (
    Normalize,
    MultiScaleRandomCrop,
    MultiScaleCornerCrop,
    Compose,
    RandomHorizontalFlip,
    ToTensor,
)
from temporal_transform import TemporalRandomCrop
from target_transform import ClassLabel
from utils import calculate_accuracy, AverageMeter


def get_opt():
    opt = parse_opts()
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = "{}-{}".format(opt.model, opt.model_depth)
    opt.mean, opt.std = get_mean_std(opt)

    return opt


def get_model(opt, path):
    assert os.path.exists(path)
    model = ResNet(
        opt, sample_duration=opt.sample_duration, sample_size=opt.sample_size
    )

    model = nn.DataParallel(model.cuda(opt.device[0]), device_ids=opt.device)

    model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
    model.module.fc = model.module.fc.cuda(opt.device[0])

    # pretrained_model = torch.load(path, map_location=lambda storage, loc: storage.cuda(opt.device[0]))
    pretrained_model = torch.load(path)
    model.load_state_dict(pretrained_model["state_dict"])
    # pretrained_model = nn.DataParallel(pretrained_model.cuda(opt.device[0]), device_ids=opt.device)
    # pretrained_model = None

    return model


def get_demo_dataloader(opt):
    print("demo~~~")
    assert opt.train_crop in ["random", "corner", "center"]
    norm_method = Normalize(opt.mean, opt.std)
    if opt.train_crop == "random":
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "corner":
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == "center":
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=["c"]
        )
    spatial_transform = Compose(
        [crop_method, RandomHorizontalFlip(), ToTensor(opt.norm_value), norm_method]
    )
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()

    training_data = demo_dataset(
        r"/data2/pingpang_dataset/videos/",
        r"/home/zhangrui/3dresnet/meta_data",
        "training",
        spatial_transform,
        temporal_transform,
        target_transform,
    )

    # for index, item in enumerate(training_data):
    #     if index % 100 == 0:
    #         print(index)
    print("success load demo_dataset")

    demo_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True,
    )

    return demo_loader


if __name__ == "__main__":
    print("hello world")
    opt = get_opt()

    model = get_model(opt, r"/home/zhangrui/my_project/results/save_30.pth")

    # criterion = nn.CrossEntropyLoss().cuda(1)
    # model = nn.DataParallel(model.cuda(1), device_ids=[1, 3])
    # model = model.cuda(1)

    cross_validation = k_folder_cross_validation(5, opt)
    train_loader, train_logger, train_batch_logger, val_loader, val_logger = (
        cross_validation()
    )

    demo_loader = get_demo_dataloader(opt)
    acc = [[] for _ in range(9)]

    # calculate accuracy for each class
    for index, (input, target) in enumerate(train_loader):
        model.eval()
        # start_time = time()
        input = Variable(input.cuda(opt.device[0]), volatile=True)
        target = Variable(target.cuda(opt.device[0]), volatile=True)

        output = model(input)

        # loss = criterion(output, target)
        acc = calculate_accuracy(output, target)

        # losses.update(loss.data[0])
        print(index, acc)

        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time:.3f}\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #     epoch,
        #     index + 1,
        #     len(dataloader),
        #     batch_time=time()-start_time,
        #     loss=losses,
        #     acc=acces))
