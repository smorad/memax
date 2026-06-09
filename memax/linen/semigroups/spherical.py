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

RotationMatrix = Float[Array, "Hidden Hidden"]
SphericalRecurrentState = RotationMatrix
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class PSphericalSemigroup(Semigroup):
    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentState:
        return jnp.eye(self.recurrent_size)

    @nn.nowrap
    def zero_carry(self) -> SphericalRecurrentState:
        return jnp.zeros((self.recurrent_size, self.recurrent_size))

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self, carry: SphericalRecurrentState, input: SphericalRecurrentState
    ) -> SphericalRecurrentState:
        return carry @ input


class PSpherical(GRAS):
    """A simple Bayesian memory model.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    hidden_size: int
    use_equinox_init: bool = True

    def setup(self):
        init = self.use_equinox_init
        self.proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = equinox_dense(
            self.proj_size, self.hidden_size, use_equinox_init=init
        )
        self.output = equinox_dense(
            self.hidden_size, self.recurrent_size, use_equinox_init=init
        )
        self.initial_vector = jnp.ones(self.recurrent_size)

    @jaxtyped(typechecker=typechecker)
    def rot(self, x) -> RotationMatrix:
        q = self.project(x)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices_from(A, 1)
        A = A.at[tri_idx].set(q)
        A = A - A.T
        R = jax.scipy.linalg.expm(A)
        return R

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        emb, start = x
        rot = self.rot(emb)
        return rot, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: SphericalRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        state, reset_carry = h

        normed = self.initial_vector / jnp.linalg.norm(self.initial_vector)
        return self.output(state @ normed)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> SphericalRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(PSphericalSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
