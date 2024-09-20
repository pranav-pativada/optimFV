import torch.nn.functional as F
import argparse
from .trainer import Trainer
from types_ import Args, Optimiser, Net, Loss, Device, DataLoader


class CIFARTrainer(Trainer):
    def __init__(
        self,
        model: Net,
        loss_fn: Loss,
        optimiser: Optimiser,
        device: Device,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ):
        super(CIFARTrainer, self).__init__(
            model, loss_fn, optimiser, device, train_loader, test_loader
        )
    
    def train(self, epoch: int) -> None:
        """
        Train the model on CIFAR for one epoch.
        """
        pass
