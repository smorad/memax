import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray, Shaped, jaxtyped

from memax.linen.gras import GRAS
from memax.linen.groups import Resettable, Semigroup
from memax.linen.inits import dense as equinox_dense
from memax.linen.scans import semigroup_scan
from memax.mtypes import Input, StartFlag

TTTRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
    Int[Array, ""],
]
TTTRecurrentStateWithReset = Tuple[TTTRecurrentState, StartFlag]


def apply_rope_at_position(
    x: Float[Array, "Feat"], position: Shaped[Array, ""]
) -> Float[Array, "Feat"]:
    """Apply RoPE to a single feature vector at a specific (1-indexed) position."""
    feat = x.shape[0]
    assert feat % 2 == 0, "Feature dimension must be even"

    theta_indices = jnp.arange(0, feat, 2, dtype=jnp.float32)
    theta = 1.0 / (10000.0 ** (theta_indices / feat))
    angle = jnp.asarray(position, dtype=jnp.float32) * theta
    rotator = jnp.exp(1j * angle)

    x_complex = x.reshape(-1, 2)
    x_complex = x_complex[..., 0] + 1j * x_complex[..., 1]
    out_complex = x_complex * rotator
    return jnp.stack([out_complex.real, out_complex.imag], axis=-1).reshape(feat)


class TTTLinearSemigroup(Semigroup):
    """The associative operation for the TTT-Linear gradient descent step."""

    recurrent_size: int
    use_rope: bool = False

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.array(0, dtype=jnp.int32),
        )

    @nn.nowrap
    def zero_carry(self) -> TTTRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.array(0, dtype=jnp.int32),
        )

    @jaxtyped(typechecker=typechecker)
    def rotate_matrix(
        self, mat: Float[Array, "Key Value"], position: Shaped[Array, ""]
    ) -> Float[Array, "Key Value"]:
        mat = jax.vmap(lambda row: apply_rope_at_position(row, position))(mat)
        mat_t = jax.vmap(lambda col: apply_rope_at_position(col, -position))(mat.T)
        return mat_t.T

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: TTTRecurrentState,
        input: TTTRecurrentState,
    ) -> TTTRecurrentState:
        # Exact same associative factorization as GDN: S_t = M_t S_{t-1} + X_t
        M_i, X_i, i = carry
        M_j, X_j, j = input

        if self.use_rope:
            # Shift the left segment by the right segment length, matching
            # the semidirect-product composition pattern used in FFM.
            M_i = self.rotate_matrix(M_i, j)
            X_i = self.rotate_matrix(X_i, j)
        return M_j @ M_i, M_j @ X_i + X_j, j + i


class TTTLinear(GRAS):
    """The minimalist Test-Time Training (Linear) architecture.

    Replaces data-dependent gating with an inner-loop gradient descent step (eta)
    to minimize a self-supervised reconstruction loss.
    """

    hidden_size: int
    recurrent_size: int
    positional_embedding: Optional[str] = None
    use_residual: bool = False  # "Undocumented" update as featured in the official code
    use_equinox_init: bool = True

    def setup(self):
        assert self.positional_embedding in [
            None,
            "rope",
        ], "positional_embedding must be None or 'rope'"
        init = self.use_equinox_init
        h, r = self.hidden_size, self.recurrent_size
        self.K = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.Q = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.V = equinox_dense(r, h, use_bias=False, use_equinox_init=init)
        self.output = equinox_dense(h, r, use_equinox_init=init)
        # Initialize the inner learning rate as a learnable parameter.
        self.eta = self.param("eta", lambda *args: jnp.array(0.1))

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentStateWithReset:
        emb, start = x

        k = self.K(emb)
        if self.positional_embedding == "rope":
            k = apply_rope_at_position(k, jnp.array(1, dtype=jnp.int32))

        v = self.V(emb)

        M = jnp.eye(self.recurrent_size) - self.eta * jnp.outer(k, k)
        if self.use_residual:
            # As implemented in the code (and not mentioned in the paper...)
            X = self.eta * jnp.outer(v - k, k)
        else:
            # As written in the paper
            X = self.eta * jnp.outer(v, k)
        dt = jnp.array(1, dtype=jnp.int32)

        return (M, X, dt), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: TTTRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (M, X, t), reset_flag = h

        q = self.Q(emb)
        if self.positional_embedding == "rope":
            q = apply_rope_at_position(q, t)

        return self.output(X @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> TTTRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(TTTLinearSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
