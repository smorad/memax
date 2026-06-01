"""Experiment configuration shared by equinox and linen training scripts."""

from dataclasses import dataclass, fields
from typing import List, Optional, Union

from memax.experiments.datasets import get_default_hyperparameters


@dataclass
class ExperimentConfig:
    seed: int = 0
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    recurrent_size: Optional[int] = None
    num_layers: Optional[int] = None
    lr: Optional[float] = None
    use_wandb: bool = False
    project_name: str = "memax-debug"
    dataset_name: str = "sequential_mnist"
    loss_function: Optional[str] = None
    models: Union[str, List[str]] = "all"
    max_train_samples: Optional[int] = None
    max_updates: Optional[int] = None
    smoke: bool = False

    @classmethod
    def from_namespace(cls, args) -> "ExperimentConfig":
        models = getattr(args, "models", None)
        if models is None:
            models = "all"
        return cls(
            seed=args.seed,
            num_epochs=getattr(args, "num_epochs", None),
            batch_size=getattr(args, "batch_size", None),
            recurrent_size=getattr(args, "recurrent_size", None),
            num_layers=getattr(args, "num_layers", None),
            lr=getattr(args, "lr", None),
            use_wandb=args.use_wandb,
            project_name=args.project_name,
            dataset_name=getattr(args, "dataset_name", "sequential_mnist"),
            loss_function=getattr(args, "loss_function", None),
            models=models,
            max_train_samples=getattr(args, "max_train_samples", None),
            max_updates=getattr(args, "max_updates", None),
            smoke=getattr(args, "smoke", False),
        )


def apply_smoke_defaults(config: ExperimentConfig) -> ExperimentConfig:
    if not config.smoke:
        return config
    config.num_epochs = 1
    config.batch_size = 2
    config.recurrent_size = 16
    config.num_layers = 1
    config.lr = 1e-3
    config.max_train_samples = config.max_train_samples or 8
    config.max_updates = config.max_updates or 2
    if config.models == "all":
        config.models = ["GRU"]
    return config


def fill_dataset_defaults(config: ExperimentConfig) -> ExperimentConfig:
    defaults = get_default_hyperparameters(config.dataset_name)
    for f in fields(config):
        if f.name in defaults and getattr(config, f.name) is None:
            setattr(config, f.name, defaults[f.name])
    return config


def prepare_config(args) -> ExperimentConfig:
    config = ExperimentConfig.from_namespace(args)
    config = apply_smoke_defaults(config)
    config = fill_dataset_defaults(config)
    return config
