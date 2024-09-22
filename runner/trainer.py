import torch
from tqdm import tqdm
from typing import Tuple
from mytypes import Tensor, DataLoader, Device, Loss, Net, Optimiser


class Trainer:
    """
    Generic trainer class for MNIST, CIFAR10, CIFAR100
    """

    def __init__(
        self,
        model: Net,
        loss_fn: Loss,
        optimiser: Optimiser,
        device: Device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        log_interval: int = 100,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.log_interval = log_interval
        self.accuracies = []

    def train(self, epoch: int) -> None:
        """
        Train the model for one epoch
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data = data.to(self.device)
            target = target.to(self.device)

            model_fn = lambda: self.model(data)
            loss_fn = lambda pred: self.loss_fn(pred, target)

            loss, predictions = self.optimise(model_fn, loss_fn)
            pred = predictions.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            accuracy = pred.eq(target.view_as(pred)).double().mean()

            if batch_idx % self.log_interval == 0:
                print(
                    "\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {:.6f}".format(
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
        Test the model for one epoch
        """
        self.model.eval()
        self.accuracies = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
                data, target = data.to(self.device), target.to(self.device)
                predictions = self.model(data)

                loss = self.loss_fn(predictions, target)
                pred = predictions.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                accuracy = pred.eq(target.view_as(pred)).double().mean()
                self.accuracies.append(accuracy)

                # log the loss and accuracy
                print(
                    "\nTest Epoch: {} [{}/{} ({:.0f}%)]\tVal Loss: {:.6f}\t Val Accuracy: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.test_loader.dataset),
                        100.0 * batch_idx / len(self.test_loader),
                        loss.item(),
                        accuracy.item(),
                    )
                )

    def optimise(self, model_fn, loss_fn) -> Tuple[Tensor, Tensor]:
        match self.optimiser.__class__.__name__:
            case "CurveBall":
                (loss, predictions) = self.optimiser.step(model_fn, loss_fn)
            case "LBFGS":
                
                def closure():
                    self.optimiser.zero_grad()
                    predictions = model_fn()
                    loss = loss_fn(predictions)
                    loss.backward()
                    return loss

                loss = self.optimiser.step(closure)
                predictions = model_fn()
            case _:
                # Standard optimisers
                self.optimiser.zero_grad()
                predictions = model_fn()
                loss = loss_fn(predictions)
                loss.backward()
                self.optimiser.step()

        return (loss, predictions)

    @property
    def accuracy(self) -> float:
        return sum(self.accuracies) / len(self.accuracies)
