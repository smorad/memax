import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, List, Optional, Tuple
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Module, Resettable, SetAction
from memax.equinox.scans import set_action_scan
from memax.mtypes import Input, StartFlag

SphericalRecurrentState = Float[Array, "Recurrent"]
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class SphericalSetAction(SetAction):
    """
    The RotRNN (recurrent update) from https://arxiv.org/abs/2407.07239

    However, this is implemented in a less efficient manner (sequential)
    """

    recurrent_size: int
    project: nn.Linear
    initial_state: jax.Array

    def __init__(self, recurrent_size: int, sequence_length: int = 1024, *, key):
        self.recurrent_size = recurrent_size
        proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = nn.Linear(recurrent_size, proj_size, key=key)
        self.initial_state = jnp.ones((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def rot(self, z: Array) -> Array:
        q = self.project(z)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices_from(A, 1)
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


class Spherical(GRAS):
    """The Spherical RNN from https://arxiv.org/abs/2407.07239

    However, this is implemented in a less efficient manner (sequential)
    than the spherical semigroup.
    """

    algebra: BinaryAlgebra
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
    recurrent_size: int
    hidden_size: int

    def __init__(self, recurrent_size, hidden_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.readout_dim = recurrent_size
        keys = jax.random.split(key)
        self.algebra = Resettable(SphericalSetAction(recurrent_size, key=keys[0]))
        self.scan = set_action_scan

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
    ) -> Float[Array, "{self.readout_dim}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)


def make_layer(hidden_size, key, **overrides):
    """Build Spherical set-action for a residual trunk.

    ``hidden_size`` is the trunk embedding width. ``recurrent_size`` (state on the
    sphere) defaults to ``hidden_size``.
    """
    return Spherical(
        hidden_size=hidden_size, recurrent_size=hidden_size, key=key, **overrides
    )
