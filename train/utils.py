import torch
import sys
import yaml
from types_ import Args, DataLoader, Device
from typing import Tuple


def parse_config(args: Args) -> Args:
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
