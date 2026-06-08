import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import Resettable, Semigroup
from memax.linen.inits import dense as equinox_dense
from memax.linen.scans import semigroup_scan
from memax.mtypes import Input, StartFlag

FWPRecurrentState = Float[Array, "Key Value"]
FWPRecurrentStateWithReset = Tuple[FWPRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FWPSemigroup(Semigroup):
    """The Additive Fast Weight Programmer semigroup (recurrent update)
    from https://arxiv.org/pdf/2508.08435"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentState:
        return jnp.zeros((self.recurrent_size, self.recurrent_size))

    @nn.nowrap
    def zero_carry(self) -> FWPRecurrentState:
        return jnp.zeros((self.recurrent_size, self.recurrent_size))

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: FWPRecurrentState,
        input: FWPRecurrentState,
    ) -> FWPRecurrentState:
        return carry + input


class FWP(GRAS):
    """The Additive Fast Weight Programmer from https://arxiv.org/pdf/2508.08435

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    use_equinox_init: bool = True

    def setup(self):
        init = self.use_equinox_init
        h, r = self.hidden_size, self.recurrent_size
        self.K = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.Q = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.V = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.output = equinox_dense(h, r, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        v = self.V(emb)
        kv = jnp.outer(k, v)
        return kv, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: FWPRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        kv_sum, reset_flag = h
        q = phi(self.Q(emb))
        return self.output(kv_sum @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> FWPRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(FWPSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
