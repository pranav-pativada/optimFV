import torch
from types_ import Args, DataLoader, Device
from typing import Tuple
from torchvision import datasets, transforms


def get_mnist_dataset(args: Args) -> Tuple[DataLoader, DataLoader]:
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    use_cuda, _ = get_device(args)

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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


def get_device(args: Args) -> Tuple[bool, Device]:
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return (use_cuda, device)
