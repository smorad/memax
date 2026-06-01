"""Shared utilities for memax training experiment scripts."""

from memax.experiments.config import (
    ExperimentConfig,
    apply_smoke_defaults,
    fill_dataset_defaults,
    prepare_config,
)
from memax.experiments.datasets import (
    DATASET_NAMES,
    get_default_hyperparameters,
    get_default_loss_name,
    load_dataset,
    make_synthetic_dataset,
    slice_dataset,
)

# Re-export sorted registry keys for CLI help / docs
from memax.experiments.runner import run_equinox_training, run_linen_training

__all__ = [
    "DATASET_NAMES",
    "ExperimentConfig",
    "apply_smoke_defaults",
    "fill_dataset_defaults",
    "prepare_config",
    "get_default_hyperparameters",
    "get_default_loss_name",
    "load_dataset",
    "make_synthetic_dataset",
    "run_equinox_training",
    "run_linen_training",
    "slice_dataset",
]
