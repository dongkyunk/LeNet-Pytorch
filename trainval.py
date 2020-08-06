from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def train(train_loader, model, criterion, optimizer, epoch, iter=0, print_freq=500):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if DEVICE == 'cuda':
        model.cuda()
    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)
        if DEVICE == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        # compute the output
        output = model(images)
        loss = criterion(output, labels)

        # record loss
        losses.update(loss.item(), images.size(0))

        # optimize && backward pass
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print(iter)
        # print(i)
        # Print logs
        if i % print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, iter+i, len(train_loader), batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg)
    print(msg)

    return iter+i, losses.avg
