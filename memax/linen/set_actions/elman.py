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

ElmanRecurrentState = Float[Array, "Recurrent"]
ElmanRecurrentStateWithReset = Tuple[ElmanRecurrentState, StartFlag]


class ElmanSetAction(SetAction):
    """
    The Elman set action

    Paper: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    activation: Callable = jax.nn.tanh

    def setup(self):
        self.U_h = nn.Dense(self.recurrent_size)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: ElmanRecurrentState, input: ElmanRecurrentState
    ) -> ElmanRecurrentState:
        return self.activation(self.U_h(carry) + input)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @nn.nowrap
    def zero_carry(self) -> ElmanRecurrentState:
        return jnp.zeros((self.recurrent_size,))


class Elman(GRAS):
    """
    The Elman Recurrent Network layer

    Paper: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    hidden_size: int
    activation: Callable = jax.nn.tanh

    def setup(self):
        self.W_h = nn.Dense(self.recurrent_size, use_bias=False)
        self.W_y = nn.Dense(self.hidden_size)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentStateWithReset:
        emb, start = x
        return self.W_h(emb), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: ElmanRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> ElmanRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(ElmanSetAction(**kwargs))

    @staticmethod
    def default_scan():
        return set_action_scan
