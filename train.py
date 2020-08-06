import numpy as np
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from utils.opt import parse_option

from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from trainval import train
from evaluate import validate
from utils.utils import save_checkpoint, get_optimizer
from model.LeNet import LeNet5

opt = parse_option()

if __name__ == '__main__':
    # download and create datasets
    train_dataset = mnist.MNIST(
        root='./train', download=True, train=True, transform=transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor()]))
    val_dataset = mnist.MNIST(root='./test', download=True, train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor()]))

    # define the data loaders
    train_loader = DataLoader(train_dataset, opt.batch_size)
    val_loader = DataLoader(val_dataset, opt.batch_size)

    model = LeNet5()
    print(model)

    optimizer = get_optimizer(opt, model)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    iter = 0
    

    for epoch in range(opt.epoch):
        # train for one epoch
        iter, loss = train(train_loader, model, criterion,
                     optimizer, epoch, iter=iter)
        # evaluate
        loss, accuracy = validate(
            val_loader, model, criterion, epoch)

        is_best = accuracy < best_accuracy
        best_accuracy = min(accuracy, best_accuracy)

        # If best_eval, best_save_path
        # Save latest/best weights in the model directory
        save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "accuracy": accuracy,
             "optimizer": optimizer.state_dict(),
             }, is_best, opt.SAVE_DIR, 'checkpoint.pth')

        print('accuracy: {:.2f}%'.format(100 * accuracy))
                        
    final_model_state_file = os.path.join(opt.SAVE_DIR, 'final_state.pth')

    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    print('Done!')

            

        
   
