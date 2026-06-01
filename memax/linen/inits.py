"""Flax parameter initializers aligned with Equinox ``nn.Linear``.

Equinox uses uniform weights and biases on ``[-1/sqrt(in_features), 1/sqrt(in_features)]``.
Flax ``nn.Dense`` defaults to LeCun normal, which can hurt training parity with Equinox models.
"""

import math
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def equinox_linear_limit(in_features: int) -> float:
    if in_features == 0:
        return 1.0
    return 1.0 / math.sqrt(in_features)


def equinox_uniform(in_features: int) -> Callable:
    """Uniform on ``[-limit, limit]`` with ``limit = 1 / sqrt(in_features)``."""

    lim = equinox_linear_limit(in_features)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-lim, maxval=lim)

    return init


def dense(
    features: int,
    in_features: int,
    *,
    use_bias: bool = True,
    use_equinox_init: bool = True,
    **kwargs,
) -> nn.Dense:
    """``nn.Dense`` with optional Equinox-compatible initialization."""
    if use_equinox_init:
        return nn.Dense(
            features,
            kernel_init=equinox_uniform(in_features),
            bias_init=equinox_uniform(in_features) if use_bias else None,
            use_bias=use_bias,
            **kwargs,
        )
    return nn.Dense(features, use_bias=use_bias, **kwargs)
