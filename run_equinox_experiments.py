"""Train recurrent memory models (Equinox) on memax datasets."""

import argparse

import jax
import wandb

from memax.equinox.train_utils import get_residual_memory_models
from memax.experiments.cli import add_experiment_args
from memax.experiments.config import prepare_config
from memax.experiments.datasets import load_dataset, slice_dataset
from memax.experiments.losses import get_equinox_loss, resolve_loss_name
from memax.experiments.runner import run_equinox_training


def main():
    parser = argparse.ArgumentParser(
        description="Train recurrent memory models (Equinox)."
    )
    add_experiment_args(parser)
    args = parser.parse_args()
    config = prepare_config(args)

    dataset = load_dataset(config.dataset_name)
    dataset = slice_dataset(dataset, config.max_train_samples)

    feature_in = dataset["x_test"].shape[-1]
    feature_out = dataset["y_test"].shape[-1]

    loss_name = resolve_loss_name(config.dataset_name, config.loss_function)
    loss_fn = get_equinox_loss(loss_name)

    key = jax.random.PRNGKey(config.seed)
    models = get_residual_memory_models(
        input=feature_in,
        hidden=config.recurrent_size,
        output=feature_out,
        num_layers=config.num_layers,
        models=config.models,
        key=key,
    )

    wandb_module = wandb if config.use_wandb else None
    for name, model in models.items():
        run_equinox_training(
            config, name, model, dataset, loss_fn, wandb_module=wandb_module
        )


if __name__ == "__main__":
    main()
