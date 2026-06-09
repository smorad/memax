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

DeltaFWPRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
DeltaFWPRecurrentStateWithReset = Tuple[DeltaFWPRecurrentState, StartFlag]


def phi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses relu
    # https://arxiv.org/pdf/2406.06484 uses silu
    return jax.nn.relu(x)


def psi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses sigmoid
    # https://arxiv.org/pdf/2508.08435 suggests 2 * sigmoid
    return 2 * jax.nn.sigmoid(x)


class DeltaNetSemigroup(Semigroup):
    """The Fast Weight Programmer w/ Delta update semigroup (recurrent update)
    from https://arxiv.org/pdf/2508.08435"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaFWPRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @nn.nowrap
    def zero_carry(self) -> DeltaFWPRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: DeltaFWPRecurrentState,
        input: DeltaFWPRecurrentState,
    ) -> DeltaFWPRecurrentState:
        # Amazing resource: https://sustcsonglin.github.io/blog/2024/deltanet-2/
        # Based on Songlin's factorization
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class DeltaNet(GRAS):
    """The Additive Fast Weight Programmer w/ Delta update from https://arxiv.org/pdf/2508.08435

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
        self.w = equinox_dense(1, h, use_equinox_init=init)
        self.output = equinox_dense(h, r, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaFWPRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        k = k / (jnp.linalg.norm(k) + 1e-6)  # normalize key
        v = self.V(emb)
        beta = psi(self.w(emb))
        M = jnp.eye(self.recurrent_size) - beta * jnp.outer(k, k)
        X = beta * jnp.outer(v, k)
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: DeltaFWPRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (M, X), reset_flag = h
        q = phi(self.Q(emb))
        q = q / (jnp.linalg.norm(q) + 1e-6)  # normalize query
        return self.output(X @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaFWPRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> DeltaFWPRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(DeltaNetSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
