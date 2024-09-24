import torch.nn as nn
from .flatten import Flatten


class Basic3C3D(nn.Module):
    """
    Basic 3C3D model for CIFAR Training
    """

    def __init__(self, num_classes: int = 10):
        super(Basic3C3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            Flatten(),
            nn.Linear(4 * 4 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
