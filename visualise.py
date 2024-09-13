import random
import torch
import matplotlib.pyplot as plt
import argparse

from torchinfo import summary
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomRotation,
    RandomResizedCrop,
)
from datetime import datetime
from argparse import Namespace
from typing import Dict, Callable
import os

activations: Dict = {}
device: torch.device = None


def get_parser() -> Namespace:
    parser = argparse.ArgumentParser(description="Feature Visualisation")
    parser.add_argument(
        "--model",
        type=str,
        default="ConvNet",
        choices=["ConvNet", "ResNet"],
        help="model to visualise",
    )
    parser.add_argument(
        "--use-hub",
        action="store_true",
        default=False,
        help="use torch hub to load the model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "ImageNet"],
        help="dataset that the model has been trained on",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="vis",
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
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    global device
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.model == "ConvNet" and args.dataset != "MNIST":
        raise ValueError("ConvNet only supports MNIST dataset")
    if args.model == "ResNet" and args.dataset != "ImageNet":
        raise ValueError("ResNet only supports ImageNet dataset")

    if not args.use_hub and args.model_path is None:
        setattr(args, "model_path", f"checkpoints/{args.model.lower()}.pt")
    if args.channel_idx == -1:
        if args.dataset == "MNIST":
            setattr(args, "channel_idx", random.randint(0, 9))  # MNIST classes
        else:
            setattr(args, "channel_idx", random.randint(0, 1000))  # ImageNet classes

    if device == torch.device("cuda"):
        setattr(args, "iters", 2048)

    return args


def plot_image(image: torch.Tensor, args: Namespace, layer: str) -> None:
    print(
        f"Visualising Model {args.model} | Layer {layer} | Channel {args.channel_idx}"
    )

    img = image.detach().cpu().squeeze(dim=0).permute(1, 2, 0)
    img = torch.clamp(img, 0, 1)
    img = plt.imshow(img, cmap="viridis")
    plt.colorbar(img)
    plt.axis("off")

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    plt.savefig(
        f"{args.vis_dir}/vis_{layer}_{args.channel_idx}_{datetime.now().strftime("%d%m%Y_%H%M%S") }.png"
    )


def get_model(args: Namespace) -> torch.nn.Module:
    model = None

    if args.use_hub:
        match args.model:
            case "ConvNet":
                raise ValueError("ConvNet not available in Torch Hub")
            case "ResNet":
                model = torch.hub.load(
                    "pytorch/vision:v0.10.0", "resnet18", pretrained=True
                )
    
    elif args.model_path:
        match args.model:
            case "ConvNet":
                from models.convnet import ConvNet

                model = ConvNet()
            case "ResNet":
                raise NotImplementedError("Please use Torch Hub to load ResNet")
            case _:
                raise ValueError(f"Model {args.model} not implemented")

        model.load_state_dict(torch.load(args.model_path))

    for param in model.parameters():
        param.requires_grad_(False)

    model.to(device)
    return model


def get_layer_hook(layer_name: str) -> Callable:
    def hook(module, input, output):
        activations[layer_name] = output

    return hook


def loss_func(layer: str, channel_idx: int) -> Callable:
    def loss() -> torch.Tensor:
        # TODO: Check this
        return -activations[layer][0, channel_idx].mean()

    return loss


def get_transforms(input_size: int) -> Compose:
    return Compose(
        [
            RandomCrop(input_size, padding=random.randint(0, 8)),  # jitter
            RandomRotation((-45, 45)),  # rotate
            RandomResizedCrop(input_size, scale=(0.9, 1.2), ratio=(1.0, 1.0)),  # scale
        ]
    )


def visualise(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_name: str,
    channel_idx: int,
    iters: int,
    transforms=None,
    log_interval: int = 500,
) -> torch.Tensor:
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(get_layer_hook(layer_name))

    model.eval()
    model(image)

    activations[layer_name] = activations[layer_name].requires_grad_().to(device)
    optimizer = torch.optim.Adam([image], lr=0.1)

    loss_fn = loss_func(layer_name, channel_idx)

    # Tweak the image
    for i in range(iters):
        transformed_image = transforms(image) if transforms else image

        optimizer.zero_grad()
        model(transformed_image)
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(f"Optimizing Image @ Iteration {i}")

    handle.remove()
    return image


def get_layer(model: torch.nn.Module) -> str:
    print("Printing Model Summary")
    stats = summary(model)
    layers = list(map(lambda x: x[0], model.named_modules()))
    print(
        "\nAvailable Layers. To visualise, please use layers which have trainable params > 0 to get meaningful results: "
    )
    for idx, layer_info in enumerate(stats.summary_list):
        if idx == 0:
            continue
        print(f"{layers[idx]}: {layer_info.num_params} trainable params")

    layer = input("\nEnter layer name to visualise: ")
    if layer not in layers:
        raise ValueError(f"Layer {layer} not found in model")

    return layer


def main():
    args = get_parser()
    torch.manual_seed(args.seed)

    model = get_model(args)
    layer = get_layer(model)
    image = (
        torch.randn(1, 1, 28, 28)
        if args.dataset == "MNIST"
        else torch.randn(1, 3, 224, 224)
    )

    transforms = get_transforms(image.shape[-1])

    image = image.detach().to(device).requires_grad_(True)
    output = visualise(model, image, layer, args.channel_idx, args.iters, transforms)
    plot_image(output, args, layer)


if __name__ == "__main__":
    main()
