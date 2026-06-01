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

SphericalRecurrentState = Float[Array, "Recurrent"]
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class SphericalSetAction(SetAction):
    """
    The RotRNN (recurrent update) from https://arxiv.org/abs/2407.07239

    However, this is implemented in a less efficient manner (sequential)
    """

    recurrent_size: int
    sequence_length: int = 1024
    use_equinox_init: bool = True

    def setup(self):
        proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = equinox_dense(
            proj_size,
            self.recurrent_size,
            use_equinox_init=self.use_equinox_init,
        )
        self.initial_state = jnp.ones((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def rot(self, z: Array) -> Array:
        q = self.project(z)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices(self.recurrent_size, 1)
        A = A.at[tri_idx].set(q)
        A = A - A.T
        R = jax.scipy.linalg.expm(A)
        return R

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: SphericalRecurrentState, input: SphericalRecurrentState
    ) -> SphericalRecurrentState:
        R = self.rot(input)
        return R @ carry

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentState:
        return self.initial_state / jnp.linalg.norm(self.initial_state)

    @nn.nowrap
    def zero_carry(self) -> SphericalRecurrentState:
        return jnp.zeros((self.recurrent_size,))


class Spherical(GRAS):
    """The Spherical RNN from https://arxiv.org/abs/2407.07239

    However, this is implemented in a less efficient manner (sequential)
    than the spherical semigroup.
    """

    recurrent_size: int
    hidden_size: int
    use_equinox_init: bool = True

    def setup(self):
        self.W_y = equinox_dense(
            self.hidden_size,
            self.recurrent_size,
            use_equinox_init=self.use_equinox_init,
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: SphericalRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

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
        return Resettable(SphericalSetAction(**kwargs))

    @staticmethod
    def default_scan():
        return set_action_scan
