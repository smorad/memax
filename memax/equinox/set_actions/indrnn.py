"""Module implementing the IndRNN cell"""
from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, SetAction, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import set_action_scan

IndRNNRecurrentState = Float[Array, "Recurrent"]
IndRNNRecurrentStateWithReset = Tuple[IndRNNRecurrentState, StartFlag]


class IndRNNMagma(SetAction):
    """
    Independently Recurrent Neural Network Cell.
    """
    recurrent_size: int
    recurrent_kernel: Array
    bias: Array
    activation: Callable
    recurrent_min_abs: float
    recurrent_max_abs: Optional[float]

    def __init__(
        self,
        recurrent_size: int,
        activation: Callable = jax.nn.relu,
        recurrent_min_abs: float = 0.0,
        recurrent_max_abs: Optional[float] = None,
        *,
        key: PRNGKeyArray,
    ):
        self.recurrent_size = recurrent_size
        self.activation = activation
        self.recurrent_min_abs = recurrent_min_abs
        self.recurrent_max_abs = recurrent_max_abs

        keys = jax.random.split(key, 2)

        self.recurrent_kernel = jax.random.uniform(
            keys[0], (recurrent_size,), minval=0.0, maxval=1.0
        )
        self.bias = jax.random.uniform(
            keys[1], (recurrent_size,), minval=0.0, maxval=1.0
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: IndRNNRecurrentState, input: IndRNNRecurrentState
    ) -> IndRNNRecurrentState:
        u = self.recurrent_kernel

        if self.recurrent_min_abs > 0:
            abs_u = jnp.abs(u)
            min_abs_u = jnp.maximum(abs_u, self.recurrent_min_abs)
            u = jnp.sign(u) * min_abs_u

        if self.recurrent_max_abs is not None:
            u = jnp.clip(u, -self.recurrent_max_abs, self.recurrent_max_abs)

        return self.activation(input + u * carry + self.bias)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IndRNNRecurrentState:
        return jnp.zeros((self.recurrent_size,))


class IndRNN(GRAS):
    """
    Independently Recurrent Neural Network (IndRNN)

    Paper: https://arxiv.org/abs/1803.04831
    """
    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [IndRNNRecurrentStateWithReset, IndRNNRecurrentStateWithReset],
                IndRNNRecurrentStateWithReset,
            ],
            IndRNNRecurrentStateWithReset,
            IndRNNRecurrentStateWithReset,
        ],
        IndRNNRecurrentStateWithReset,
    ]
    recurrent_size: int
    hidden_size: int


    def __init__(
        self,
        recurrent_size: int,
        hidden_size: int,
        activation: Callable = jax.nn.relu,
        recurrent_min_abs: float = 0.0,
        recurrent_max_abs: Optional[float] = None,
        max_timesteps: int = 1024,
        *,
        key: PRNGKeyArray,
    ):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        keys = jax.random.split(key, 3)

        if recurrent_max_abs is None and max_timesteps is not None:
             recurrent_max_abs = 2.0 ** (1.0 / max_timesteps)

        self.algebra = Resettable(
            IndRNNMagma(
                recurrent_size,
                activation=activation,
                recurrent_min_abs=recurrent_min_abs,
                recurrent_max_abs=recurrent_max_abs,
                key=keys[0],
            )
        )
        self.scan = set_action_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IndRNNRecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: IndRNNRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IndRNNRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
