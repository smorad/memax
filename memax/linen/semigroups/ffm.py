import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray, Real, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import Resettable, Semigroup
from memax.linen.inits import dense as equinox_dense
from memax.linen.scans import semigroup_scan
from memax.mtypes import Input, StartFlag

FFMRecurrentState = Tuple[Complex[Array, "Trace Context"], Int[Array, ""]]
FFMRecurrentStateWithReset = Tuple[FFMRecurrentState, StartFlag]


class FFMSemigroup(Semigroup):
    """The Fast and Forgetful Memory semigroup (recurrent update) from https://arxiv.org/abs/2310.04128."""

    trace_size: int
    context_size: int
    deterministic_init: bool = True
    min_period: float = 1
    max_period: float = 1024

    def setup(self):
        if self.deterministic_init:
            a_low = 1e-6
            a_high = 0.5
            a = jnp.linspace(a_low, a_high, self.trace_size)
            b = (
                2
                * jnp.pi
                / jnp.linspace(self.min_period, self.max_period, self.context_size)
            )
            self.params = (a, b)
        else:
            raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    def log_gamma(self, t: Real[Array, ""]) -> Complex[Array, "Trace Context"]:
        a, b = self.params
        a = -jnp.abs(a).reshape((self.trace_size, 1))
        b = b.reshape(1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(1, 1)

    @jaxtyped(typechecker=typechecker)
    def gamma(self, t: Real[Array, ""]) -> Complex[Array, "Trace Context"]:
        return jnp.exp(self.log_gamma(t))

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentState:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        carry_shape = (self.trace_size, self.context_size)

        return jnp.zeros(carry_shape, dtype=jnp.complex64), jnp.array(
            0, dtype=jnp.int32
        )

    @nn.nowrap
    def zero_carry(self) -> FFMRecurrentState:
        return (
            jnp.zeros((self.trace_size, self.context_size), dtype=jnp.complex64),
            jnp.array(0, dtype=jnp.int32),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self, carry: FFMRecurrentState, input: FFMRecurrentState
    ) -> FFMRecurrentState:
        (
            state,
            i,
        ) = carry
        x, j = input
        state = state * self.gamma(j) + x
        return state, j + i


class FFM(GRAS):
    """Fast and Forgetful Memory from https://arxiv.org/abs/2310.04128."""

    hidden_size: int
    trace_size: int
    context_size: int
    use_equinox_init: bool = True

    def setup(self):
        init = self.use_equinox_init
        h = self.hidden_size
        self.pre = equinox_dense(self.trace_size, h, use_equinox_init=init)
        self.gate_in = equinox_dense(self.trace_size, h, use_equinox_init=init)
        self.gate_out = equinox_dense(h, h, use_equinox_init=init)
        self.mix = equinox_dense(
            h, 2 * self.trace_size * self.context_size, use_equinox_init=init
        )
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentStateWithReset:
        emb, start = x
        gate_in = jax.nn.sigmoid(self.gate_in(emb))
        pre = self.pre(emb)
        gated = pre * gate_in
        scan_input = jnp.repeat(
            jnp.expand_dims(gated, 1), self.context_size, axis=1
        ).astype(jnp.complex64)
        dt = jnp.array(1)
        return (scan_input, dt), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: FFMRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        (z, dt), reset_flag = h
        emb, start = x
        z = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1).reshape(-1)
        z = self.mix(z)
        gate_out = jax.nn.sigmoid(self.gate_out(emb))
        out = self.ln(z * gate_out) + emb * (1 - gate_out)
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> FFMRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(FFMSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
