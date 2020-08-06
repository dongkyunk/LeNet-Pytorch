import torch
import torch.nn as nn
from collections import OrderedDict

# parameters
n_classes = 10


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layerC1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=(5, 5), stride=1),
            nn.Tanh()
        )

        self.layerS2 = nn.AvgPool2d(kernel_size=2)

        self.layerC3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=(5, 5), stride=1),
            nn.Tanh()
        )

        self.layerS4 = nn.AvgPool2d(kernel_size=2)

        self.layerC5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120,
                      kernel_size=(5, 5), stride=1),
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
        # print(x.size())
        x = self.layerS2(x)
        # print(x.size())
        x = self.layerC3(x)
        # print(x.size())
        x = self.layerS4(x)
        # print(x.size())
        x = self.layerC5(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.layerF6(x)
        # print(x.size())
        x = self.outputLayer(x)

        return x
