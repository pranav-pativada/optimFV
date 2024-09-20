import torch.nn as nn
from .flatten import Flatten


class ConvNet(nn.Module):
    """
    Basic ConvNet model for MNIST Training
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(),
            Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.layers(x)
