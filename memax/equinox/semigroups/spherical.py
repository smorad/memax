import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, List, Optional, Tuple
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Resettable, Semigroup
from memax.equinox.scans import semigroup_scan
from memax.mtypes import Input, StartFlag

RotationMatrix = Float[Array, "Hidden Hidden"]
SphericalRecurrentState = RotationMatrix
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class PSphericalSemigroup(Semigroup):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentState:
        return jnp.eye(self.recurrent_size)

    @jaxtyped(typechecker=typechecker)
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
    proj_size: int
    scan: Callable[
        [
            Callable[
                [SphericalRecurrentStateWithReset, SphericalRecurrentStateWithReset],
                SphericalRecurrentStateWithReset,
            ],
            SphericalRecurrentStateWithReset,
            SphericalRecurrentStateWithReset,
        ],
        SphericalRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra
    initial_vector: jax.Array
    project: nn.Linear

    def __init__(self, recurrent_size, hidden_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.readout_dim = recurrent_size
        keys = jax.random.split(key)
        self.proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = nn.Linear(hidden_size, self.proj_size, key=keys[0])
        self.algebra = Resettable(PSphericalSemigroup(recurrent_size))
        self.scan = semigroup_scan
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
    ) -> Float[Array, "{self.readout_dim}"]:
        emb, start = x
        state, reset_carry = h

        normed = self.initial_vector / jnp.linalg.norm(self.initial_vector)
        return state @ normed

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
