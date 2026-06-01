import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import BinaryAlgebra, Resettable, SetAction
from memax.linen.inits import dense as equinox_dense
from memax.linen.scans import set_action_scan
from memax.mtypes import Input, StartFlag

GRURecurrentState = Float[Array, "Recurrent"]
GRURecurrentStateWithReset = Tuple[GRURecurrentState, StartFlag]


class GRUSetAction(SetAction):
    """
    The Gated Recurrent Unit set action

    Paper: https://arxiv.org/abs/1406.1078
    """

    recurrent_size: int
    use_equinox_init: bool = True

    def setup(self):
        n = self.recurrent_size
        init = self.use_equinox_init
        self.U_z = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_r = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_h = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.W_z = equinox_dense(n, n, use_equinox_init=init)
        self.W_r = equinox_dense(n, n, use_equinox_init=init)
        self.W_h = equinox_dense(n, n, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: GRURecurrentState, input: Float[Array, "Recurrent"]
    ) -> GRURecurrentState:
        z = jax.nn.sigmoid(self.W_z(input) + self.U_z(carry))
        r = jax.nn.sigmoid(self.W_r(input) + self.U_r(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(r * carry))
        out = (1 - z) * carry + z * h_hat
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @nn.nowrap
    def zero_carry(self) -> GRURecurrentState:
        return jnp.zeros((self.recurrent_size,))


class GRU(GRAS):
    """
    The Gated Recurrent Unit

    Paper: https://arxiv.org/abs/1406.1078
    """

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: GRURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> GRURecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(GRUSetAction(**kwargs))

    @staticmethod
    def default_scan():
        return set_action_scan
