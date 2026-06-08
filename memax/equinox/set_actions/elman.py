import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Callable, Optional, Tuple
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.gras import GRAS
from memax.equinox.groups import BinaryAlgebra, Resettable, SetAction
from memax.equinox.scans import set_action_scan
from memax.mtypes import Input, StartFlag

ElmanRecurrentState = Float[Array, "Recurrent"]
ElmanRecurrentStateWithReset = Tuple[ElmanRecurrentState, StartFlag]


class ElmanSetAction(SetAction):
    """
    The Elman set action

    Paper: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    U_h: nn.Linear
    activation: eqx.Module

    def __init__(self, recurrent_size: int, activation=jax.nn.tanh, *, key):
        self.recurrent_size = recurrent_size
        self.U_h = nn.Linear(recurrent_size, recurrent_size, key=key)
        self.activation = activation

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: ElmanRecurrentState, input: ElmanRecurrentState
    ) -> ElmanRecurrentState:
        return self.activation(self.U_h(carry) + input)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentState:
        return jnp.zeros((self.recurrent_size,))


class Elman(GRAS):
    """
    The Elman Recurrent Network layer

    Paper: https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [ElmanRecurrentStateWithReset, ElmanRecurrentStateWithReset],
                ElmanRecurrentStateWithReset,
            ],
            ElmanRecurrentStateWithReset,
            ElmanRecurrentStateWithReset,
        ],
        ElmanRecurrentStateWithReset,
    ]
    recurrent_size: int
    hidden_size: int
    W_h: nn.Linear

    def __init__(self, recurrent_size, hidden_size, activation=jax.nn.tanh, *, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.readout_dim = recurrent_size
        keys = jax.random.split(key, 2)
        self.algebra = Resettable(
            ElmanSetAction(recurrent_size, activation=activation, key=keys[0])
        )
        self.scan = set_action_scan
        self.W_h = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentStateWithReset:
        emb, start = x
        return self.W_h(emb), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: ElmanRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.readout_dim}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> ElmanRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)


def make_layer(hidden_size, key, *, activation=jax.nn.tanh, **overrides):
    """Build Elman for a residual trunk.

    ``hidden_size`` is the trunk embedding width. ``recurrent_size`` (state length)
    defaults to ``hidden_size``. Use ``activation=jax.nn.relu`` for the ``ElmanReLU``
    registry entry.
    """
    return Elman(
        hidden_size=hidden_size,
        recurrent_size=hidden_size,
        key=key,
        activation=activation,
        **overrides,
    )
