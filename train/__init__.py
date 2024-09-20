# __init__.py
from .trainer import Trainer as Trainer
from .mnist import MNISTrainer as MNISTrainer
from .cifar import CIFARTrainer as CIFARTrainer
from .utils import get_mnist_dataset as get_mnist_dataset, get_device
