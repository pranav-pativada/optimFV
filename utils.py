import torch
from torchvision import datasets, transforms
from mytypes import Args, Device, DataLoader, Net, Optimiser, Tensor
from typing import Tuple, Dict, List
from models import ConvNet, Basic3C3D
from optimisers import CurveBall


def get_data(args: Args, use_cuda: bool) -> Tuple[Net, DataLoader, DataLoader]:
    if args.dataset == "MNIST":
        train_loader, test_loader = get_mnist(args, use_cuda)
    else:
        train_loader, test_loader = get_cifar(args, use_cuda)

    return (train_loader, test_loader)


def get_model(args: Args, device: Device) -> Net:
    match args.model:
        case "ConvNet":
            model = ConvNet()
        case "Basic3C3D":
            if args.dataset == "CIFAR10":
                model = Basic3C3D(num_classes=10)
            else:
                model = Basic3C3D(num_classes=100)
        case "ResNet":
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=True
            )
        case _:
            raise ValueError(f"{args.model} not supported.")

    if getattr(args, "model_path", None):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from {args.model_path} with accuracy @")
        for param in model.parameters():
            param.requires_grad_(False)

    if getattr(args, "parallel", False):
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    return model


def get_optimiser(args: Args, params: List) -> Optimiser:
    # Default values for optimiser arguments are set to that for training
    match args.optimiser:
        case "AdaGrad": 
            args.lr = 0.01 if args.lr < 0 else args.lr
            return torch.optim.Adagrad(params, lr=args.lr)
        case "Adam":
            args.lr = 0.001 if args.lr < 0 else args.lr
            return torch.optim.Adam(params, lr=args.lr)
        case "SGD":
            args.lr = 0.01 if args.lr < 0 else args.lr
            args.momentum = 0.9 if args.momentum < 0 else args.momentum
            return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
        case "LBFGS":
            args.lr = 1 if args.lr < 0 else args.lr
            return torch.optim.LBFGS(params, lr=args.lr)
        case "CurveBall":
            args.lr = 0.01 if args.lr < 0 else args.lr
            args.momentum = 0.9 if args.momentum < 0 else args.momentum
            lambd = 1.0 if args.lambd < 0 else args.lambd
            return CurveBall(
                params,
                lr=args.lr,
                momentum=args.momentum,
                lambd=lambd,
                auto_lambda=not args.no_auto_lambda,
            )
        case _:
            raise ValueError(f"Unknown optimiser: {args.optimiser}")


def get_cifar(args: Args, use_cuda: bool) -> Tuple[DataLoader, DataLoader]:
    train_kwargs, test_kwargs = get_train_test_args(args, use_cuda)
    Dataset = datasets.CIFAR10 if args.dataset == "CIFAR10" else datasets.CIFAR100

    train_loader = torch.utils.data.DataLoader(
        Dataset(
            args.datadir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        **train_kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        Dataset(
            args.datadir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        **test_kwargs,
    )

    return (train_loader, test_loader)


def get_mnist(args: Args, use_cuda: bool) -> Tuple[DataLoader, DataLoader]:
    train_kwargs, test_kwargs = get_train_test_args(args, use_cuda)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.datadir,
            train=True,
            download=True,
            transform=transform,
        ),
        **train_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.datadir,
            train=False,
            transform=transform,
        ),
        **test_kwargs,
    )

    return (train_loader, test_loader)


def get_train_test_args(args: Args, use_cuda: bool) -> Tuple[Dict, Dict]:
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    return (train_kwargs, test_kwargs)


def get_device(args: Args) -> Tuple[bool, Device]:
    """
    Get the device to use for training
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return (use_cuda, device)


def get_gaussian_image(args: Args) -> Tensor:
    match args.dataset:
        case "MNIST":
            return torch.randn(1, 1, 28, 28)
        case "CIFAR10" | "CIFAR100":
            return torch.randn(1, 3, 32, 32)
        case "ImageNet":
            return torch.randn(1, 3, 224, 224)
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
