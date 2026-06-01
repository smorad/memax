import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import Resettable, SetAction
from memax.linen.inits import dense as equinox_dense
from memax.linen.scans import set_action_scan
from memax.mtypes import Input, StartFlag

MGURecurrentState = Float[Array, "Recurrent"]
MGURecurrentStateWithReset = Tuple[MGURecurrentState, StartFlag]


class MGUSetAction(SetAction):
    """
    The Minimal Gated Unit set action

    Paper: https://arxiv.org/abs/1701.03452
    """

    recurrent_size: int
    use_equinox_init: bool = True

    def setup(self):
        n = self.recurrent_size
        init = self.use_equinox_init
        self.U_h = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_f = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.W_h = equinox_dense(n, n, use_equinox_init=init)
        self.W_f = equinox_dense(n, n, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: MGURecurrentState, input: Float[Array, "Recurrent"]
    ) -> MGURecurrentState:
        f = jax.nn.sigmoid(self.W_f(input) + self.U_f(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(f * carry))
        h = (1 - f) * carry + f * h_hat
        return h

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @nn.nowrap
    def zero_carry(self) -> MGURecurrentState:
        return jnp.zeros((self.recurrent_size,))


class MGU(GRAS):
    """The Minimal Gated Unit layer

    Paper: https://arxiv.org/abs/1701.03452
    """

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: MGURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> MGURecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(MGUSetAction(**kwargs))

    @staticmethod
    def default_scan():
        return set_action_scan
