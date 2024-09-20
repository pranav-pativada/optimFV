import argparse
import os
import sys
import yaml
from types_ import Args


class Parser:
    def __init__(self):
        args_ = self._get()
        self.parse_config(args_)
        self.validate_args(args_)
        self.args_ = args_

    @staticmethod
    def _get() -> Args:
        parser = argparse.ArgumentParser(description="MNIST Training")
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
            choices=["SGD", "Adam", "L-BFGS", "CurveBall"],
            default="Adam",
            help="optimiser (SGD, Adam, L-BFGS, or CurveBall)",
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
            help="learning rate (default: 0.01 for SGD, 0.001 for Adam, 1 for L-BFGS and CurveBall)",
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=-1,
            metavar="M",
            help="momentum (default: 0.9 for SGD and CurveBall)",
        )
        parser.add_argument("--lambda", type=float, default=1.0, help="lambda")
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
            default="./experiments",
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

    def parse_config(self, args: Args):
        if (
            args.config
            and len(sys.argv) == 3
            and sys.argv[1] == "-config"
            and sys.argv[2].endswith(".yaml")
        ):
            data = None
            with open(args.config, "r") as file:
                data = yaml.safe_load(file)

            for key, value in data.items():
                key = key.replace("-", "_")  # Namespace representation converts - to _
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    raise ValueError(f"Invalid entry {key} in yaml file")

        elif args.config and len(sys.argv) > 3:
            raise ValueError(
                "Please specify either a valid config file OR the required arguments."
            )

    def validate_args(self, args: Args):
        if args.dataset == "MNIST" and args.model == "Basic3C3D":
            raise ValueError("Basic3C3D is not supported for MNIST")
        if args.dataset != "MNIST" and args.model == "ConvNet":
            raise ValueError("ConvNet not supported for CIFAR10 and CIFAR100")

        if not args.datadir:
            args.datadir = os.path.join("./data", args.dataset)

        if not args.outputdir:
            args.outputdir = os.path.join("experiments", args.dataset)
        if not os.path.exists(args.outputdir):
            os.makedirs(args.outputdir)

    @property
    def args(self):
        return self.args_
