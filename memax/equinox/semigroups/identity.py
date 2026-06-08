import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Resettable, Semigroup
from memax.equinox.scans import semigroup_scan
from memax.mtypes import Input, StartFlag

IdentityRecurrentState = Float[Array, "0"]
IdentityRecurrentStateWithReset = Tuple[IdentityRecurrentState, StartFlag]


class IdentitySemigroup(Semigroup):
    """Trivial semigroup with empty state (no memory)."""

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IdentityRecurrentState:
        return jnp.zeros((0,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: IdentityRecurrentState, input: IdentityRecurrentState
    ) -> IdentityRecurrentState:
        return jnp.zeros((0,))


class Identity(GRAS):
    """A memory-free recurrent cell for controlled baselines.

    This module uses the same :class:`~memax.equinox.gras.GRAS` and
    :class:`~memax.equinox.models.residual.ResidualModel` interfaces as the other
    cells, but its algebra carries no information across time steps. Each timestep's
    readout features are simply the current layer input embedding.

    When stacked in a residual trunk, all learnable mixing happens in ``map_in``,
    per-layer :class:`~memax.equinox.models.layer_mixer.LayerMixer` blocks, and
    ``map_out``—not in recurrence. Use this to compare sequence models against a
    memory-free baseline with an identical training and evaluation setup.
    """

    recurrent_size: int
    readout_dim: int
    scan: Callable[
        [
            Callable[
                [
                    IdentityRecurrentStateWithReset,
                    IdentityRecurrentStateWithReset,
                ],
                IdentityRecurrentStateWithReset,
            ],
            IdentityRecurrentStateWithReset,
            IdentityRecurrentStateWithReset,
        ],
        IdentityRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    def __init__(self, recurrent_size, key):
        del key
        self.recurrent_size = recurrent_size
        self.readout_dim = recurrent_size
        self.algebra = Resettable(IdentitySemigroup())
        self.scan = semigroup_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IdentityRecurrentStateWithReset:
        _, start = x
        return (jnp.zeros((0,)), start)

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: IdentityRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.readout_dim}"]:
        emb, start = x
        _, _reset_carry = h
        return emb

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> IdentityRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)


def make_layer(hidden_size: int, key, **overrides):
    """Build Identity for a residual trunk (memory-free baseline).

    ``hidden_size`` is the trunk embedding width and readout dimension.
    """
    return Identity(
        recurrent_size=hidden_size,
        key=key,
        **overrides,
    )
