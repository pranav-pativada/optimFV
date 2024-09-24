import os
import argparse
from .parser import Parser
from mytypes import Args


class VisParser(Parser):
    def __init__self():
        super().__init__()

    def _get(self) -> Args:
        parser = argparse.ArgumentParser(description="Activation Maximisation")
        parser.add_argument(
            "--interactive",
            type=bool,
            default=False,
            help="enable interactive mode",
        )
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            metavar="PATH",
            help="path to the config file",
        )
        parser.add_argument(
            "--optim-train", 
            type=str, 
            default="Adam",
            help="optimiser that the model was trained on",
        )
        parser.add_argument(
            "--optimiser",
            type=str,
            default="Adam",
            choices=["Adam", "SGD", "AdaGrad", "LBFGS"],
            help="optimiser to use",
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
        parser.add_argument(
            "--model",
            type=str,
            default="ConvNet",
            choices=["ConvNet", "Basic3C3D", "ResNet"],
            help="model to visualise",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="MNIST",
            choices=["MNIST", "CIFAR10", "CIFAR100", "ImageNet"],
            help="dataset that the model has been trained on",
        )
        parser.add_argument(
            "--use-hub",
            action="store_true",
            default=False,
            help="use pretrained models from torch.hub",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            metavar="PATH",
            help="path to the model",
        )
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
            "--seed",
            type=int,
            default=42,
            metavar="S",
            help="random seed (default: 42)",
        )
        parser.add_argument(
            "--vis-dir",
            type=str,
            default=None,
            metavar="DIR",
            help="directory to save visualisations",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=1024,
            metavar="N",
            help="number of iterations to run the optimisation",
        )
        parser.add_argument(
            "--channel-idx",
            type=int,
            default=0,
            metavar="N",
            help="channel index to visualise in the layer",
        )
        parser.add_argument(
            "--layer",
            type=str,
            default=None,
            metavar="NAME",
            help="layer name to visualise",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=100,
            metavar="N",
            help="optimisation logging interval",
        )
        args = parser.parse_args()
        return args

    def _validate_args(self, args: Args) -> None:
        if args.dataset == "MNIST" and args.model != "ConvNet":
            raise ValueError("MNIST only supported for ConvNet")
        elif args.dataset.startswith("CIFAR") and args.model != "Basic3C3D":
            raise ValueError("CIFAR only supported for 3C3D")
        elif args.dataset == "ImageNet" and args.model != "ResNet":
            raise ValueError("ImageNet only supported for ResNet")

        if (args.use_hub and args.model_path) or (
            not args.use_hub and not args.model_path
        ):
            raise ValueError("Please specify either model path or use hub")

        if args.model_path and not args.vis_dir:
            args.vis_dir = os.path.join("vis", args.dataset, args.optim_train)
        elif args.use_hub and not args.vis_dir:
            if args.optimiser == "SGD" and args.momentum == 0: 
                args.vis_dir = os.path.join("vis", args.dataset, args.optimiser, "no_momentum")
            else: 
                args.vis_dir = os.path.join("vis", args.dataset, args.optimiser)

        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
