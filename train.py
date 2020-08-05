from model.LeNet import LeNet5
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from trainval import train
from evaluate import validate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32

if __name__ == '__main__':
    # download and create datasets
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    # define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LeNet5()
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _epoch in range(epoch):

        iter = train(train_loader, model, criterion,
                     optimizer, epoch, iter=iter)
        # evaluate
        nme, predictions = validate(
            val_loader, model, criterion, epoch, num_joint=args.num_joint)

        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            sgd.step()

        correct = 0
        _sum = 0

        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))