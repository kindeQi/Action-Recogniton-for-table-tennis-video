import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
from dataset import UCF101, pingpong_dataset
from utils import Logger
from spatial_transform import Normalize, MultiScaleRandomCrop, MultiScaleCornerCrop, Compose, RandomHorizontalFlip, ToTensor
from temporal_transform import TemporalRandomCrop
from target_transform import ClassLabel
from utils import calculate_accuracy, AverageMeter

def get_trainingset(opt, dataset='UCF101'):
    print('training~~~')
    assert opt.train_crop in ['random', 'corner', 'center']
    norm_method = Normalize(opt.mean, opt.std)
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()

    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform,
            temporal_transform,
            target_transform)
        print('success load ucf101')

    elif opt.dataset == 'pingpong_dataset':
        training_data = pingpong_dataset(
            r'/data/pingpang_dataset/videos/',
            r'/home/zhangrui/3dresnet/meta_data',
            'training',
            spatial_transform,
            temporal_transform,
            target_transform)

        # for index, item in enumerate(training_data):
        #     if index % 100 == 0:
        #         print(index)
        print('success load pingpong_dataset')

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)

    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    return train_loader, train_logger, train_batch_logger


def train_network(opt, model, criterion, optimizer, train_logger, dataloader, epoch=1):

    # batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    print('lr: {}'.format(optimizer.param_groups[-1]['lr']))

    model.train()
    for index, (input, target) in enumerate(dataloader):
        start_time = time()

        input = Variable(input.cuda(opt.device[0]))
        target = Variable(target.cuda(opt.device[0]))

        output = model(input)
        loss = criterion(output, target)
        acc = calculate_accuracy(output, target, top_n=1)

        losses.update(loss.data[0])
        acces.update(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

        start_time = time()

    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': acces.avg,
        'lr': optimizer.param_groups[-1]['lr']})

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

# if __name__ == '__main__':
