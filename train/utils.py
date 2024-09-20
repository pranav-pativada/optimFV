import torch
from torchvision import datasets, transforms
from types_ import Args, Device, DataLoader, Net
from typing import Tuple, Dict
from models import ConvNet, Basic3C3D


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
