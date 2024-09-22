import argparse
import os
from .parser import Parser
from mytypes import Args

class TrainParser(Parser):
    def __init__self():
        super().__init__()
    
    def _get(self) -> Args:
        parser = argparse.ArgumentParser(description="Example training of small models")
        parser.add_argument(
            "--model",
            type=str,
            default="ConvNet",
            choices=["ConvNet", "Basic3C3D"],
            help="model to use (ConvNet or 3C3D)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="MNIST",
            choices=["MNIST", "CIFAR10", "CIFAR100"],
            help="dataset to use (MNIST, CIFAR10, or CIFAR100)",
        )
        parser.add_argument(
            "--optimiser",
            choices=["SGD", "Adam", "LBFGS", "CurveBall"],
            default="Adam",
            help="optimiser (SGD, Adam, LBFGS, or CurveBall)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=1000,
            help="input batch size for testing (default: 1000 for MNIST, 64 for CIFAR)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="number of epochs to train (default: 10)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=-1,
            metavar="LR",
            help="learning rate (default: 0.01 for SGD, 0.001 for Adam, 1 for LBFGS and CurveBall)",
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=-1,
            metavar="M",
            help="momentum (default: 0.9 for SGD and CurveBall)",
        )
        parser.add_argument("--lambd", type=float, default=1.0, help="lambda")
        parser.add_argument(
            "--no-auto-lambda",
            action="store_true",
            default=False,
            help="disables automatic lambda estimation",
        )
        parser.add_argument("--no-batch-norm", action="store_true", default=False)
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--no-mps",
            action="store_true",
            default=False,
            help="disables macOS GPU training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            metavar="FILE",
            help="YAML config file",
        )
        parser.add_argument("--datadir", type=str, default=None, help="Data directory")
        parser.add_argument(
            "--outputdir",
            type=str,
            default=None,
            help="output directory",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=100,
            metavar="N",
            help="Interval to log",
        )
        parser.add_argument(
            "--save-model",
            action="store_true",
            default=True,
            help="Saves the current model after training",
        )
        parser.add_argument(
            "--save-interval",
            type=int,
            default=10,
            metavar="N",
            help="Interval to save model",
        )
        parser.add_argument(
            "--parallel", action="store_true", default=False, help="Parallel training"
        )

        args = parser.parse_args()
        return args
    
    def _validate_args(self, args: Args) -> None:
        if args.dataset == "MNIST" and args.model == "Basic3C3D":
            raise ValueError("Basic3C3D is not supported for MNIST")
        if args.dataset != "MNIST" and args.model == "ConvNet":
            raise ValueError("ConvNet not supported for CIFAR10 and CIFAR100")

        if not args.datadir:
            args.datadir = "./data"

        if not args.outputdir:
            args.outputdir = os.path.join("experiments", args.dataset, args.optimiser)
        if not os.path.exists(args.outputdir):
            os.makedirs(args.outputdir)