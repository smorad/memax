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

LSTMRecurrentState = Tuple[Float[Array, "Recurrent"], Float[Array, "Recurrent"]]
LSTMRecurrentStateWithReset = Tuple[LSTMRecurrentState, StartFlag]


class LSTMSetAction(SetAction):
    """
    The Long Short-Term Memory set action

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    recurrent_size: int
    use_equinox_init: bool = True

    def setup(self):
        n = self.recurrent_size
        init = self.use_equinox_init
        self.U_f = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_i = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_o = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.U_c = equinox_dense(n, n, use_bias=False, use_equinox_init=init)
        self.W_f = equinox_dense(n, n, use_equinox_init=init)
        self.W_i = equinox_dense(n, n, use_equinox_init=init)
        self.W_o = equinox_dense(n, n, use_equinox_init=init)
        self.W_c = equinox_dense(n, n, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LSTMRecurrentState, input: LSTMRecurrentState
    ) -> LSTMRecurrentState:
        x_t, _ = input
        c, h = carry
        f_f = jax.nn.sigmoid(self.W_f(x_t) + self.U_f(h))
        f_i = jax.nn.sigmoid(self.W_i(x_t) + self.U_i(h))
        f_o = jax.nn.sigmoid(self.W_o(x_t) + self.U_o(h))
        f_c = jax.nn.tanh(self.W_c(x_t) + self.U_c(h))

        c = f_f * c + f_i * f_c
        h = f_o * jax.nn.tanh(c)

        return (c, h)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,)),
        )

    @nn.nowrap
    def zero_carry(self) -> LSTMRecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,)),
        )


class LSTM(GRAS):
    """
    The Long Short-Term Memory layer

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentStateWithReset:
        emb, start = x
        c = jnp.zeros_like(emb)
        return (emb, c), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LSTMRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "Recurrent"]:
        (c_t, h_t), reset_flag = h
        emb, start = x
        return h_t

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> LSTMRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(LSTMSetAction(**kwargs))

    @staticmethod
    def default_scan():
        return set_action_scan
