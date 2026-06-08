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

DeltaProductRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
DeltaProductRecurrentStateWithReset = Tuple[DeltaProductRecurrentState, StartFlag]


def phi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses relu
    # https://arxiv.org/pdf/2406.06484 and
    # https://arxiv.org/pdf/2502.10297 use silu
    return jax.nn.silu(x)


def psi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses sigmoid
    # https://arxiv.org/pdf/2508.08435 suggests 2 * sigmoid
    return 2 * jax.nn.sigmoid(x)


class DeltaProductSemigroup(Semigroup):
    """The Delta Product update semigroup (recurrent update)
    from https://arxiv.org/abs/2502.10297"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaProductRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @nn.nowrap
    def zero_carry(self) -> DeltaProductRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: DeltaProductRecurrentState,
        input: DeltaProductRecurrentState,
    ) -> DeltaProductRecurrentState:
        # Amazing resource: https://sustcsonglin.github.io/blog/2024/DeltaProduct-2/
        # Based on Songlin's factorization
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class DeltaProduct(GRAS):
    """The Delta Product w/ Delta update from https://arxiv.org/abs/2502.10297

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    rank: int
    use_equinox_init: bool = True

    def setup(self):
        init = self.use_equinox_init
        h, r = self.hidden_size, self.recurrent_size
        self.K = equinox_dense(r * self.rank, h, use_bias=False, use_equinox_init=init)
        self.Q = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.V = equinox_dense(r * self.rank, h, use_bias=False, use_equinox_init=init)
        self.w = equinox_dense(self.rank, h, use_equinox_init=init)
        self.alpha = equinox_dense(1, h, use_equinox_init=init)
        self.output = equinox_dense(h, r, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaProductRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb)).reshape(-1, self.rank)
        k = k / (jnp.linalg.norm(k) + 1e-6)  # normalize key
        v = self.V(emb).reshape(-1, self.rank)
        alpha = jax.nn.sigmoid(self.alpha(emb))
        beta = psi(self.w(emb)).reshape(-1, self.rank)

        beta_outer = lambda u, v, beta: beta * jnp.outer(u, v)

        # Outer returns tensor of (rank, recurrent, recurrent), sum reduces to (recurrent, recurrent)
        M = alpha * (
            jnp.eye(self.recurrent_size)
            - jnp.prod(jax.vmap(beta_outer, in_axes=-1)(k, k, beta), axis=0)
        )
        X = jax.vmap(beta_outer, in_axes=-1)(v, k, beta).sum(axis=0)
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: DeltaProductRecurrentStateWithReset,
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
    ) -> DeltaProductRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> DeltaProductRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(DeltaProductSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
