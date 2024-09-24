import torch.nn as nn
from .flatten import Flatten


class Basic3C3D(nn.Module):
    """
    Basic 3C3D model for CIFAR Training
    """

    def __init__(self, num_classes: int = 10):
        super(Basic3C3D, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
