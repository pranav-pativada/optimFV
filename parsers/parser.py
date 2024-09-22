import sys
import yaml
from mytypes import Args
from abc import ABC, abstractmethod


class Parser(ABC):
    def __init__(self):
        args_ = self._get()
        self._parse_config(args_)
        self._validate_args(args_)
        self.args_ = args_

    @abstractmethod
    def _get(self) -> Args:
        pass

    def _parse_config(self, args: Args) -> None:
        if (
            args.config
            and len(sys.argv) == 3
            and sys.argv[1] == "--config"
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

    @abstractmethod
    def _validate_args(self, args: Args) -> None:
        pass

    @property
    def args(self):
        return self.args_
