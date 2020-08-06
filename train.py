from model.LeNet import LeNet5
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from trainval import train
from evaluate import validate
from utils import save_checkpoint
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
SAVE_DIR = 'weights'


if __name__ == '__main__':
    # download and create datasets
    train_dataset = mnist.MNIST(
        root='./train', train=True, transform=ToTensor())
    val_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    # define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LeNet5()
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    max_epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(epoch):
        # train for one epoch
        iter = train(train_loader, model, criterion,
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
             }, is_best, SAVE_DIR, 'checkpoint.pth')

        print('accuracy: {:.2f}'.format(accuracy))

    final_model_state_file = os.path.join(SAVE_DIR, 'final_state.pth')

    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)