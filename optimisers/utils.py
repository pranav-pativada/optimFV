import torch
from optimisers import CurveBall
from types_ import Args, Net, Optimiser


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
        case "Curveball":
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
