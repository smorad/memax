"""Dataset loading for memax experiment scripts."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from memax.datasets.continuous_localization import get_rot_dataset, get_trans_dataset
from memax.datasets.hub import (
    CONTINUOUS_LOCALIZATION_SEQ_LENS,
    MNIST_MATH_SEQ_LENS,
    continuous_localization_hub_id,
    mnist_math_hub_id,
)
from memax.datasets.mnist_math import get_dataset as get_mnist_math
from memax.datasets.sequential_mnist import get_dataset as get_sequential_mnist

# canonical_name -> defaults (loss + hyperparameters use canonical keys only)
_CANONICAL_LOSS = {
    "sequential_mnist": "loss_classify_terminal_output",
    "mnist_math": "loss_classify_terminal_output",
    "continuous_localization_rotation": "loss_regress_terminal_output",
    "continuous_localization": "loss_regress_terminal_output",
}

_CANONICAL_HYPERPARAMETERS = {
    "sequential_mnist": {
        "num_epochs": 5,
        "batch_size": 16,
        "recurrent_size": 256,
        "num_layers": 2,
        "lr": 1e-4,
    },
    "mnist_math": {
        "num_epochs": 5,
        "batch_size": 16,
        "recurrent_size": 256,
        "num_layers": 2,
        "lr": 1e-4,
    },
    "continuous_localization_rotation": {
        "num_epochs": 5,
        "batch_size": 16,
        "recurrent_size": 256,
        "num_layers": 2,
        "lr": 1e-4,
    },
    "continuous_localization": {
        "num_epochs": 5,
        "batch_size": 16,
        "recurrent_size": 256,
        "num_layers": 2,
        "lr": 1e-4,
    },
}

# CLI / Hub name -> (canonical task, kwargs for the loader)
_DATASET_REGISTRY: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "sequential_mnist": ("sequential_mnist", {}),
    # MNIST Math (default length 100)
    "mnist_math": ("mnist_math", {"seq_len": 100}),
    # Continuous localization (default length 20)
    "continuous_localization": (
        "continuous_localization",
        {"sequence_length": 20},
    ),
    "continuous_localization_rotation": (
        "continuous_localization_rotation",
        {"sequence_length": 20},
    ),
    "sequential_rotation": (
        "continuous_localization_rotation",
        {"sequence_length": 20},
    ),  # legacy alias
}

# Register every Hub MNIST Math repo id and friendly CLI names.
for _seq_len in MNIST_MATH_SEQ_LENS:
    _entry = ("mnist_math", {"seq_len": _seq_len})
    _DATASET_REGISTRY[f"mnist-math-{_seq_len}"] = _entry
    _DATASET_REGISTRY[mnist_math_hub_id(_seq_len)] = _entry

# Register every Hub continuous-localization repo id and friendly CLI names.
for _length in CONTINUOUS_LOCALIZATION_SEQ_LENS:
    _full = ("continuous_localization", {"sequence_length": _length})
    _rot = ("continuous_localization_rotation", {"sequence_length": _length})
    _DATASET_REGISTRY[f"continuous-localization-{_length}"] = _full
    _DATASET_REGISTRY[continuous_localization_hub_id(_length)] = _full
    _DATASET_REGISTRY[f"continuous-localization-rotation-{_length}"] = _rot

DATASET_NAMES = tuple(sorted(_DATASET_REGISTRY.keys()))


def get_default_loss_name(dataset_name: str) -> str:
    canonical, _ = _resolve_name(dataset_name)
    if canonical not in _CANONICAL_LOSS:
        raise ValueError(f"No default loss for dataset: {dataset_name}")
    return _CANONICAL_LOSS[canonical]


def get_default_hyperparameters(dataset_name: str) -> Dict[str, Any]:
    canonical, _ = _resolve_name(dataset_name)
    if canonical not in _CANONICAL_HYPERPARAMETERS:
        raise ValueError(f"No default hyperparameters for dataset: {dataset_name}")
    return dict(_CANONICAL_HYPERPARAMETERS[canonical])


def _resolve_name(dataset_name: str) -> Tuple[str, Dict[str, Any]]:
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from: {', '.join(DATASET_NAMES)}"
        )
    canonical, extra = _DATASET_REGISTRY[dataset_name]
    return canonical, dict(extra)


def load_dataset(dataset_name: str, **kwargs) -> Dict[str, Any]:
    canonical, extra = _resolve_name(dataset_name)
    opts = {**extra, **kwargs}
    if canonical == "sequential_mnist":
        return get_sequential_mnist()
    if canonical == "mnist_math":
        return get_mnist_math(seq_len=opts.get("seq_len", 100))
    if canonical == "continuous_localization_rotation":
        return get_rot_dataset(
            sequence_length=opts.get("sequence_length", 20),
        )
    if canonical == "continuous_localization":
        return get_trans_dataset(
            sequence_length=opts.get("sequence_length", 20),
        )
    raise ValueError(f"No loader for canonical dataset: {canonical}")


def slice_dataset(
    dataset: Dict[str, Any], max_train_samples: Optional[int]
) -> Dict[str, Any]:
    if max_train_samples is None:
        return dataset
    n = min(max_train_samples, dataset["size"])
    out = dict(dataset)
    out["x_train"] = dataset["x_train"][:n]
    out["y_train"] = dataset["y_train"][:n]
    out["size"] = n
    return out


def make_synthetic_dataset(
    task: str,
    *,
    num_train: int = 8,
    time: int = 16,
    key: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """Small in-memory dataset for smoke tests (no Hugging Face)."""
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if task == "classify":
        feature_in, feature_out, num_labels = 1, 10, 10
        x = jax.random.uniform(k1, (num_train, time, feature_in))
        labels = jax.random.randint(k2, (num_train,), 0, num_labels)
        y = jax.nn.one_hot(labels, num_labels)
    elif task == "regress":
        feature_in, feature_out = 3, 6
        num_labels = feature_out
        x = jax.random.normal(k1, (num_train, time, feature_in)) * 0.1
        y = jax.random.normal(k2, (num_train, feature_out)) * 0.1
    else:
        raise ValueError(f"Unknown synthetic task: {task}")

    x_test = jax.random.uniform(k3, (4, time, feature_in))
    if task == "classify":
        test_labels = jax.random.randint(k4, (4,), 0, num_labels)
        y_test = jax.nn.one_hot(test_labels, num_labels)
    else:
        y_test = jax.random.normal(k4, (4, feature_out)) * 0.1

    return {
        "x_train": x,
        "y_train": y,
        "x_test": x_test,
        "y_test": y_test,
        "num_labels": num_labels,
        "size": num_train,
    }
