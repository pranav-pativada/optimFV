import torch
import torch.nn.functional as F
from torchinfo import summary
from parser import Parser
from trainer import Trainer
from utils import get_device, get_model_and_data
from optimisers import get_optimiser


def main() -> None:
    args = Parser().args
    torch.manual_seed(args.seed)
    use_cuda, device = get_device(args)

    model, train_loader, test_loader = get_model_and_data(args, use_cuda, device)
    optimiser = get_optimiser(args, model)

    summary(model, input_size=(args.batch_size, 1, 28, 28))

    trainer = Trainer(
        model=model,
        loss_fn=F.cross_entropy,
        optimiser=optimiser,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)

        if epoch % args.save_interval == 0:
            print(f"Saving model at epoch {epoch}")
            state = {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch,
                "acc": trainer.accuracy,
            }
            torch.save(state, f"{args.outputdir}/convnet_epoch_{epoch}.pt")

        if trainer.accuracy > best_acc:
            print(f"Saving best model at epoch {epoch}")
            state = {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "epoch": epoch,
                "acc": trainer.accuracy,
            }
            torch.save(state, f"{args.outputdir}/convnet_best.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"{args.outputdir}/mnist_cnn.pt")


if __name__ == "__main__":
    main()