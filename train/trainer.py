import sys
import yaml
from abc import ABC, abstractmethod
from typing import Tuple
from types_ import Tensor, Args, DataLoader, Device, Loss, Net, Optimiser


class Trainer(ABC):
    def __init__(
        self,
        model: Net,
        loss_fn: Loss,
        optimiser: Optimiser,
        device: Device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        log_interval: int = 100,
        save_interval: int = 10,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.log_interval = log_interval
        self.save_interval = save_interval

    @abstractmethod
    def train(self, epoch: int) -> None:
        pass

    @abstractmethod
    def test(self, epoch: int) -> None:
        pass

    def optimise(self, model_fn, loss_fn) -> Tuple[Tensor, Tensor]:
        match self.optimizer.__class__.__name__:
            case "CurveBall":
                (loss, predictions) = self.optimizer.step(model_fn, loss_fn)
            case "L-BFGS":

                def closure():
                    self.optimizer.zero_grad()
                    predictions = model_fn()
                    loss = loss_fn(predictions)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)
            case _:
                # Standard optimisers
                self.optimizer.zero_grad()
                predictions = model_fn()
                loss = loss_fn(predictions)
                loss.backward()
                self.optimizer.step()

        return (loss, predictions)
