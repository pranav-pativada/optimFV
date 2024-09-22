import torch

from parsers import VisParser
from utils import get_device, get_model, get_gaussian_image, get_optimiser

from runner import Visualiser


def main():
    args = VisParser().args
    torch.manual_seed(args.seed)
    use_cuda, device = get_device(args)

    model = get_model(args, device)
    image = get_gaussian_image(args, device)
    optimiser = get_optimiser(args, [image])
    visualiser = Visualiser(
        args=args,
        model=model,
        optimiser=optimiser,
        device=device,
        use_transforms=True,
    )
    visualiser.set_transforms(image.shape[-1])
    visualiser.print_info()
    visualiser.activation_maximise(image)
    visualiser.visualise()


if __name__ == "__main__":
    main()
