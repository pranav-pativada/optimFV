import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten module
    """

    def forward(self, input):
        return input.view(input.size(0), -1)
