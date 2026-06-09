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

GDNRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
GDNRecurrentStateWithReset = Tuple[GDNRecurrentState, StartFlag]


def phi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses relu
    # https://arxiv.org/pdf/2406.06484 uses silu
    return jax.nn.silu(x)


def psi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses sigmoid
    # https://arxiv.org/pdf/2508.08435 suggests 2 * sigmoid
    return 2 * jax.nn.sigmoid(x)


def alpha_bias_init(key, shape, dtype=jnp.float32):
    return jnp.full(shape, 4.0, dtype=dtype)


class GDNSemigroup(Semigroup):
    """The Gated Delta Net semigroup (recurrent update)
    from https://arxiv.org/pdf/2412.06464"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GDNRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @nn.nowrap
    def zero_carry(self) -> GDNRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: GDNRecurrentState,
        input: GDNRecurrentState,
    ) -> GDNRecurrentState:
        # Amazing resource: https://sustcsonglin.github.io/blog/2024/deltanet-2/
        # Based on Songlin's factorization
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class GDN(GRAS):
    """The Gated Delta Network from https://arxiv.org/pdf/2412.06464

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
        # Initialize alpha bias to 4.0 so that sigmoid(alpha) is near 1.0 at init
        self.alpha = equinox_dense(
            1, h, bias_init=alpha_bias_init, use_equinox_init=init
        )
        self.output = equinox_dense(h, r, use_equinox_init=init)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GDNRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        k = k / (jnp.linalg.norm(k) + 1e-6)  # normalize key
        v = self.V(emb)
        beta = psi(self.w(emb))
        alpha = jax.nn.sigmoid(self.alpha(emb))
        M = alpha * (jnp.eye(self.recurrent_size) - beta * jnp.outer(k, k))
        X = beta * jnp.outer(v, k)
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: GDNRecurrentStateWithReset,
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
    ) -> GDNRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> GDNRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(GDNSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
