"""Train recurrent memory models (Flax Linen) on memax datasets."""

import argparse

import jax
import wandb

from memax.experiments.cli import add_experiment_args
from memax.experiments.config import prepare_config
from memax.experiments.datasets import load_dataset, slice_dataset
from memax.experiments.losses import resolve_loss_name
from memax.experiments.runner import run_linen_training
from memax.linen.train_utils import get_residual_memory_models, make_linen_loss_fn


def main():
    parser = argparse.ArgumentParser(
        description="Train recurrent memory models (Linen)."
    )
    add_experiment_args(parser)
    args = parser.parse_args()
    config = prepare_config(args)

    dataset = load_dataset(config.dataset_name)
    dataset = slice_dataset(dataset, config.max_train_samples)

    feature_in = dataset["x_test"].shape[-1]
    feature_out = dataset["y_test"].shape[-1]

    loss_name = resolve_loss_name(config.dataset_name, config.loss_function)
    models = get_residual_memory_models(
        hidden=config.recurrent_size,
        output=feature_out,
        num_layers=config.num_layers,
        models=config.models,
        model_kwargs={"input": feature_in},
    )

    wandb_module = wandb if config.use_wandb else None
    for name, model in models.items():
        loss_fn = make_linen_loss_fn(model, loss_name)
        run_linen_training(
            config, name, model, dataset, loss_fn, wandb_module=wandb_module
        )


if __name__ == "__main__":
    main()
