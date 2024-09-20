import torch
from torchvision import datasets, transforms
from types_ import Args, Device, DataLoader, Net, Optimiser
from typing import Tuple, Dict
from models import ConvNet, Basic3C3D
from optimisers import CurveBall


def get_model_and_data(
    args: Args, use_cuda: bool, device: Device
) -> Tuple[Net, DataLoader, DataLoader]:
    if args.dataset == "MNIST":
        train_loader, test_loader = get_mnist(args, use_cuda)
        model = ConvNet().to(device)
    else:
        train_loader, test_loader = get_cifar(args, use_cuda)
        model = Basic3C3D().to(device)
    if args.parallel:
        model = torch.nn.DataParallel(model)

    return (model, train_loader, test_loader)

def get_optimiser(args: Args, net: Net) -> Optimiser:
    match args.optimiser:
        case "Adam":
            args.lr = 0.001 if args.lr < 0 else args.lr
            return torch.optim.Adam(net.parameters(), lr=args.lr)
        case "SGD":
            args.lr = 0.01 if args.lr < 0 else args.lr
            args.momentum = 0.9 if args.momentum < 0 else args.momentum
            return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        case "L-BFGS":
            args.lr = 1 if args.lr < 0 else args.lr
            return torch.optim.LBFGS(net.parameters(), lr=args.lr)
        case "CurveBall":
            args.lr = 0.01 if args.lr < 0 else args.lr
            args.momentum = 0.9 if args.momentum < 0 else args.momentum
            lambd = 1.0 if args.lambd < 0 else args.lambd
            return CurveBall(
                net.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                lambd=lambd,
                auto_lambda=not args.auto_lambda,
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
