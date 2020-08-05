import torch
import torch.nn as nn
from collections import OrderedDict

#parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
n_classes = 10

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.layerC1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5,5), stride= 1),
            nn.Tanh()
        )

        self.layerS2 = nn.AvgPool2d(kernel_size=2)

        self.layerC3 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5,5), stride= 1),
            nn.Tanh()
        )        

        self.layerS4 = nn.AvgPool2d(kernel_size=2)

        self.layerC5 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5,5), stride= 1),
            nn.Tanh()
        )    
                
        self.layerF6 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(in_features=84, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layerC1(x)
        x = self.layerS2(x)
        x = self.layerC3(x)
        x = self.layerS4(x)
        x = self.layerC5(x)
        x = self.layerF6(x)
        x = self.outputLayer(x)

        return x


