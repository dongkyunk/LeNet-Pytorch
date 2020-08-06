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
    accuracy = 0.0
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (image, labels) in enumerate(val_loader):
            # measure data time
            data_time.update(time.time() - end)

            # move to GPU if available
            if DEVICE == 'cuda':
                image = image.cuda()
                labels = labels.cuda()

            # compute the output
            output = model(image)
            loss = criterion(output, labels)

            # record loss (average is updated)
            losses.update(loss.item(), image.size(0))

            # convert 'output' tensor to numpy array and use 'numpy.argmax()' function    
            output = output.cpu().data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
            output = np.argmax(output, axis = 1)  # retrieved max_values along every row    

            # measure accuracy        
            accuracy += (output == labels).sum()
            accuracy = accuracy / labels.size()
 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} accuracy:{:.4f}'\
        .format(epoch, batch_time.avg, 100.0*losses.avg, 100.0*accuracy)
    print(msg)

    return losses.avg, accuracy

