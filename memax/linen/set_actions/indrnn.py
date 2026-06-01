"""Module implementing the IndRNN cell"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import Resettable, SetAction
from memax.linen.scans import set_action_scan
from memax.mtypes import Input, StartFlag

IndRNNRecurrentState = Float[Array, "Recurrent"]
IndRNNRecurrentStateWithReset = Tuple[IndRNNRecurrentState, StartFlag]


class IndRNNSetAction(SetAction):
    """
    Independently Recurrent Neural Network Cell.
    """

    recurrent_size: int
    activation: Callable = jax.nn.relu
    recurrent_min_abs: float = 0.0
    recurrent_max_abs: Optional[float] = None
    use_equinox_init: bool = True

    def setup(self):
        def uniform_0_1(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype, minval=0.0, maxval=1.0)

        self.recurrent_kernel = self.param(
            "recurrent_kernel", uniform_0_1, (self.recurrent_size,)
        )
        self.bias = self.param("bias", uniform_0_1, (self.recurrent_size,))

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

    @nn.nowrap
    def zero_carry(self) -> IndRNNRecurrentState:
        return jnp.zeros((self.recurrent_size,))


class IndRNN(GRAS):
    """
    Independently Recurrent Neural Network (IndRNN)

    Paper: https://arxiv.org/abs/1803.04831
    """

    recurrent_size: int
    hidden_size: int
    activation: Callable = jax.nn.relu
    recurrent_min_abs: float = 0.0
    recurrent_max_abs: Optional[float] = None
    max_timesteps: int = 1024

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

    @nn.nowrap
    def zero_carry(self) -> IndRNNRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(
        recurrent_size: int,
        activation: Callable = jax.nn.relu,
        recurrent_min_abs: float = 0.0,
        recurrent_max_abs: Optional[float] = None,
        max_timesteps: int = 1024,
        **kwargs,
    ):
        if recurrent_max_abs is None and max_timesteps is not None:
            recurrent_max_abs = 2.0 ** (1.0 / max_timesteps)
        return Resettable(
            IndRNNSetAction(
                recurrent_size=recurrent_size,
                activation=activation,
                recurrent_min_abs=recurrent_min_abs,
                recurrent_max_abs=recurrent_max_abs,
                **kwargs,
            )
        )

    @staticmethod
    def default_scan():
        return set_action_scan
