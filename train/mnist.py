import torch
import argparse
import torch.nn.functional as F
from .trainer import Trainer
from types_ import Args, Optimiser, Net, Loss, Device, DataLoader


class MNISTrainer(Trainer):
    def __init__(
        self,
        model: Net,
        loss_fn: Loss,
        optimiser: Optimiser,
        device: Device,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ):
        super(MNISTrainer, self).__init__(
            model, loss_fn, optimiser, device, train_loader, test_loader
        )

    def train(self, epoch: int) -> None:
        """
        Train the model on MNIST for one epoch
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            model_fn = lambda: self.model(data)
            loss_fn = lambda pred: F.cross_entropy(pred, target)

            loss, predictions = self.optimise(model_fn, loss_fn)
            pred = predictions.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            accuracy = pred.eq(target.view_as(pred)).double().mean()

            if batch_idx % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {::6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                        accuracy.item(),
                    )
                )

    def test(self, epoch: int) -> None:
        """
        Test the model on MNIST
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                predictions = self.model(data)

                loss = F.cross_entropy(predictions, target)

                pred = predictions.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                accuracy = pred.eq(target.view_as(pred)).double().mean()

                # log the loss and accuracy
                if batch_idx % self.log_interval == 0:
                    print(
                        "Test Epoch: {} [{}/{} ({:.0f}%)]\tVal Loss: {:.6f}\t Val Accuracy: {::6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(self.test_loader.dataset),
                            100.0 * batch_idx / len(self.test_loader),
                            loss.item(),
                            accuracy.item(),
                        )
                    )
