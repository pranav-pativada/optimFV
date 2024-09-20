import torch
import typing
from argparse import Namespace

# Parser aliases
type Args = Namespace 

# Torch aliases
type Tensor = torch.Tensor
type Optimiser = torch.optim.Optimizer
type Net = torch.nn.Module
type Loss = typing.Callable[[torch.Tensor], torch.Tensor]
type Device = torch.device

# Dataset aliases
type DataLoader = torch.utils.data.DataLoader