from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
from adamp import AdamP


def get_optimizer(opt, model, momentum=0.9, wd=5e-4, nesterov=False):
    optimizer = None
    if opt.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=nesterov
        )
    elif opt.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr
        )
    elif opt.optim == 'adamp':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    return optimizer


def save_checkpoint(state, is_best, file_path, file_name='checkpoint.pth'):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    savepath = os.path.join(file_path, file_name)
    if not os.path.exists(file_path):
        print("Save Directory does not exist! Making directory {}".format(file_path))
        os.mkdir(file_path)
    else:
        print("Save Directory exists! ")
    torch.save(state, savepath)
    # Save best accuracy model weights in the model directory
    if is_best:
        savepath = os.path.join(file_path, 'best.pth')
        torch.save(state, savepath)
