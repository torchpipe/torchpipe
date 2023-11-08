# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# torchpipe in train 项目
# @ Author:  LeoSpring , Kirby_Star
# This demo provide example which use torchpipe to preprocess datas in gpu during training phase.
# By this method,you can align train-deploy results in gpu_decode to utilize it's high concurrent processing capability.
# 目的: 将torchpipe引入到pytorch训练中，进而达到对齐gpu解码augment操作，进而可以在infer中使用gpu解码部署。
# 本代码作为 dp 训练 的demo
# 只需在你的ddp代码中，按照step 1 - 7 修改即可。

from torchpipe.tool.gpu_train_tools import cv2_loader, Dataloader, TensorToTensor
import argparse
import os
import random
import shutil
import time
import warnings
import sys
#

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFile
import cv2
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

cudnn.benchmark = True

# step 1: import library


def get_parse():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        help='model architecture')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=True,
                        help='use pre-trained model')
    parser.add_argument('--gpus', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('--num-classes', default=0,
                        type=int, help='number of classes')

    parser.add_argument('--pretrained_model', default="./models/resnet101.pth",
                        type=str, help="pretrained model path")

    parser.add_argument('--output_checkpoint_path',
                        default="./checkpoint_resnet101_size_224_scene_class_6_second_version.pth.tar", type=str, help="output checkpoint path")

    parser.add_argument('--best_checkpoint_path', default="./model_best_resnet101_size_224_scene_class_6_second_version.pth.tar",
                        type=str, help="output best checkpoint path")

    parser.add_argument('--weight_decay_schedules', default="20,30",
                        type=str, help="epoch for learning rate decay")
    args = parser.parse_args()
    return args


best_acc1 = 0


class cv2Resize(object):

    def __init__(self, size=[]):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.size)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main_worker(args.gpus,  args)


def main_worker(gpus, args):
    global best_acc1
    args.gpus = gpus

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        print("number of classes : {:}".format(args.num_classes))
        print("weight_decay_schedules:{:}".format(args.weight_decay_schedules))

        if args.arch == 'resnet50':
            import timm
            model = timm.create_model('resnet50', pretrained=True)
            # model.load_state_dict(torch.load(args.pretrained_model))
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, args.num_classes)

    if args.gpus is not None:
        model = torch.nn.DataParallel(
            model, device_ids=[int(index) for index in args.gpus.split(",")])
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # download dataset
    if not os.path.exists(args.data) and args.data == 'hymenoptera_data':
        os.system("wget https://download.pytorch.org/tutorial/hymenoptera_data.zip")
        os.system("unzip hymenoptera_data.zip")
        os.remove('hymenoptera_data.zip')

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # step 2: dataset not need augment, and change loader to cv2_loader
    train_dataset = datasets.ImageFolder(traindir, loader=cv2_loader)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    print(train_dataset.class_to_idx)

    # step 3: 设置 gpu augment , 这里不需要resize操作，resize在torchpipe的toml里面设置了，
    # 只设置其他的就行，与原始的pytorch唯一不同的是 [ToTensor] 变成了自定义的 [TensorToTensor]
    train_transform_gpu = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.05),
        transforms.RandomGrayscale(0.02),
        transforms.RandomRotation(10),
        # 最后一个hue在gpu运算会比较慢(1080ti, 其他显卡不会)
        transforms.ColorJitter(0.05, 0.05, 0.05),
        TensorToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # step 4: 如果要做gpu与cpu的联合预处理，需要同时设置 cpu augment, 这个就是正常按照pytorch原来的就行。
    train_transform_cpu = transforms.Compose([
        cv2Resize((args.image_size, args.image_size)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.05),
        transforms.RandomGrayscale(0.02),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # step 5: 将dataloader类进行包装，包装成我们的Dataloader类。
    wrap_train_loader = Dataloader(
        parallel_type='dp',
        dataloader=train_loader,
        toml_path='./toml/gpu_decode_train.toml',
        transforms_gpu=train_transform_gpu,
        transforms_cpu=train_transform_cpu,
        cpu_percentage=0
    )

    # step 6: 将val做跟train同样的操作。

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, loader=cv2_loader),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    val_transform_gpu = transforms.Compose([
        TensorToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    wrap_val_loader = Dataloader(
        parallel_type='dp',
        dataloader=val_loader,
        toml_path='./toml/gpu_decode_val.toml',
        transforms_gpu=val_transform_gpu,
        transforms_cpu=None,
        cpu_percentage=0
    )

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        # step 7: 每个epoch需要重置一下迭代器
        wrap_train_loader.reset()
        train(wrap_train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # step 7: 每个epoch需要重置一下迭代器
        wrap_val_loader.reset()
        acc1 = validate(wrap_val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, filename=args.output_checkpoint_path, best_filename=args.best_checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for index, (input, target) in enumerate(train_loader):
        try:
            data_time.update(time.time() - end)
            input = input.cuda()
            target = target.cuda()

        # compute output
            output = model(input)
            loss = criterion(output, target)
        except Exception as e:
            print(e)
            continue

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Speed: {3} samples/s\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, index, len(train_loader), int(1.0/batch_time.val*args.batch_size), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_iterator = iter(val_loader)
        for i in range(len(val_loader)):
            try:
                (input, target) = next(val_iterator)
                input = input.cuda()
                target = target.cuda()

            # compute output
                output = model(input)
                loss = criterion(output, target)
            except Exception as e:
                print(e)
                continue

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename, best_filename):

    pd_path = filename[:filename.rfind('/')]
    if not os.path.exists(pd_path):
        os.makedirs(pd_path)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= int(args.weight_decay_schedules.split(",")[0]):
        lr = args.lr
    elif epoch <= int(args.weight_decay_schedules.split(",")[1]):
        lr = args.lr*0.1
    else:
        lr = args.lr*0.01
    print("learning_rate:{:}".format(str(lr)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main(get_parse())
