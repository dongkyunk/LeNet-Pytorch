from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


import torch
import numpy as np

def train(train_loader, model, criterion, optimizer, epoch, iter=0, print_freq=10):
    model.train()
    loss = 0

    for i, (inp, target, meta) in enumerate(train_loader):

        # compute the output
        output = model(inp)
        loss = critertion(output, target)

        # optimize && backward pass
        optimizer.zero_grad() # clear previous gradients
        loss.backward() # compute gradients of all variables wrt loss
        optimizer.step() # perform updates using calculated gradients

        losses.update(100.0*loss.item(), inp.size(0))

        # Print logs
        if i % print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, iter+i, len(train_loader), loss=losses)
            print(msg)


    msg = 'Train Epoch {}, loss:{:.4f} nme:{:.4f}'\
        .format(epoch, losses.avg)
    print(msg)
    return iter+i




