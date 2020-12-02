import os
import argparse
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    p = argparse.ArgumentParser(description='')

    # Data Directory
    p.add_argument('--dataset', default='', type=str)
    p.add_argument('--test_dataset', default='', type=str)

    # Input image
    p.add_argument('--img_size', default=32, type=int)

    # Optimizer
    p.add_argument('--optim', default='sgd', type=str,
                   help='RMSprop | SGD | Adam | AdamW')
    p.add_argument('--lr', default=0.001, type=float, help='learning rate')

    # Hyper-parameter
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--epoch', default=15, type=int)

    # Loss function
    p.add_argument('--criterion', default='Cross Entropy',
                   type=str, help='loss')

    # Resume trained network
    p.add_argument('--resume', default='False',
                   type=bool, help="resuming or not")
    p.add_argument('--resume_path', default='', type=str, help="pth file path")

    # Output directory
    p.add_argument('--SAVE_DIR', default='weights', type=str)

    opt = p.parse_args()

    return opt
