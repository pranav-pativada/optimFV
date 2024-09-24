import torch
import matplotlib.pyplot as plt
from mytypes import Net, Device, Optimiser, Args, Tensor
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomRotation,
    RandomResizedCrop,
)
from torchinfo import summary
from tqdm import tqdm


class Visualiser:
    def __init__(
        self,
        args: Args,
        model: Net,
        optimiser: Optimiser,
        device: Device,
        log,
        use_transforms: bool = True,
    ):
        self.model = model
        self.optimiser = optimiser
        self.device = device
        self.iters = args.iters
        self.interactive = args.interactive
        self.layer = args.layer
        self.channel_idx = args.channel_idx
        self.vis_dir = args.vis_dir
        self.activations = {}

        def loss_fn(layer_name: str, channel_idx: int) -> Tensor:
            return -self.activations[layer_name][0, channel_idx].mean()

        def get_layer_hook(layer_name: str):
            def hook(module, input, output):
                self.activations[layer_name] = output

            return hook

        self.loss_fn = loss_fn
        self.get_layer_hook = get_layer_hook

    def print_info(self) -> None:
        stats = summary(self.model)  # no forward pass
        layers = list(map(lambda x: x[0], self.model.named_modules()))
        if self.interactive:
            print("\nAvailable Layers: ")
            for idx, layer_info in enumerate(stats.summary_list):
                if idx == 0:
                    continue
                print(f"{layers[idx]}: {layer_info.num_params} trainable params")

            self.layer = input("\nEnter layer name to visualise: ")
            self.channel_idx = int(input("\nEnter channel index to maximise: "))

        if self.layer is None or self.layer not in layers:
            raise ValueError(f"Layer {self.layer} not found in model")

    def activation_maximise(self, image: Tensor) -> Tensor:
        layer = getattr(self.model, self.layer)
        handle = layer.register_forward_hook(self.get_layer_hook(self.layer))

        self.model.eval()
        self.model(image)
        self.activations[self.layer] = (
            self.activations[self.layer].requires_grad_().to(self.device)
        )

        loss_fn = lambda: self.loss_fn(self.layer, self.channel_idx)

        for i in tqdm(range(self.iters), desc="Optimising Image", unit="step"):
            transformed_image = self.transforms(image) if self.use_transforms else image

            self.optimizer.zero_grad()
            self.model(transformed_image)
            loss = loss_fn()
            loss.backward()
            self.optimizer.step()

        handle.remove()
        return image

    def visualise(self, image: Tensor) -> None:
        model_name = self.model.__class__.__name__
        print(
            f"Visualising Layer {self.layer} in model {model_name} @ channel {self.channel_idx}"
        )

        img = image.detach().cpu().squeeze(dim=0).permute(1, 2, 0)
        img = torch.clamp(img, 0, 1)
        img = plt.imshow(img, cmap="viridis")
        plt.colorbar(img)
        plt.axis("off")

        plt.savefig(f"{self.vis_dir}/{model_name}_{self.layer}_{self.channel_idx}.png")

    def set_transforms(self, image_size: int) -> None:
        if self.use_transforms:
            self.transforms = Compose(
                [
                    RandomCrop(image_size),
                    RandomRotation(10),
                    RandomResizedCrop(image_size),
                ]
            )
