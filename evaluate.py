from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trainval import AverageMeter
import time
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):
            # measure data time
            data_time.update(time.time() - end)

            inp = inp.cuda()
            target = target.cuda()

            # compute the output
            output = model(image)
            loss = criterion(output, target)

            # record loss
            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f}' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, 100.0*losses.avg)
    print(msg)

    return nme, predictions