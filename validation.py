import os
import torch
from time import time
from torch.autograd import Variable

from spatial_transform import Normalize, Compose, Scale, CenterCrop, ToTensor,MultiScaleRandomCrop
from temporal_transform import LoopPadding
from target_transform import ClassLabel
from dataset import UCF101
from utils import Logger, calculate_accuracy, AverageMeter


def get_validationset(opt):
    print('validating~~~~')
    norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        # Mu(opt.scales, opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = ClassLabel()
    validation_data = UCF101(
        root_path=opt.video_path,
        annotation_path=opt.annotation_path,
        subset='validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        n_samples_for_each_video=opt.n_val_samples,
        sample_duration=opt.sample_duration
    )
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
    return val_loader, val_logger


def validate_network(opt, model, dataloader, val_log, criterion, epoch=1):

    model.eval()

    losses = AverageMeter()
    acces = AverageMeter()

    acc_per_class = [0] * 9
    gt_per_class = [0] * 9

    for index, (input, target) in enumerate(dataloader):
        start_time = time()
        input = Variable(input, volatile=True)
        target = Variable(target.cuda(opt.device[0]), volatile=True)

        output = model(input)

        loss = criterion(output, target)
        acc = calculate_accuracy(output, target)

        losses.update(loss.data[0])
        acces.update(acc)

        # just to output the accuracy per class
        _, topk = output.topk(1, dim=1)
        for i, item in enumerate(target.data):

            t = topk.cpu().data[i]
            gt_per_class[int(t)] += 1

            if (item in t) == True:
                acc_per_class[int(t)] += 1


        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time:.3f}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch,
            index + 1,
            len(dataloader),
            batch_time=time()-start_time,
            loss=losses,
            acc=acces))

    # print(acc_per_class)
    # print(gt_per_class)

    for i in range(9):
        if gt_per_class[i] == 0:
            acc_per_class[i] = 0
        else:
            acc_per_class[i] = 1. * acc_per_class[i] / gt_per_class[i]
    # print(acc_per_class)


    val_log.log({'epoch': epoch, 'loss': losses.avg, 'acc': acces.avg})

    return losses.avg

