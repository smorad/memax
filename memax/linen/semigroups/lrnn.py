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

LinearRNNRecurrentState = Float[Array, "Hidden"]
LinearRNNRecurrentStateWithReset = Tuple[LinearRNNRecurrentState, StartFlag]


class LinearRNNSemigroup(Semigroup):
    """A simple (associative) linear recurrence"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LinearRNNRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @nn.nowrap
    def zero_carry(self) -> LinearRNNRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self, carry: LinearRNNRecurrentState, input: LinearRNNRecurrentState
    ) -> LinearRNNRecurrentState:
        return carry + input


class LinearRecurrent(GRAS):
    """A simple Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    use_equinox_init: bool = True

    def setup(self):
        init = self.use_equinox_init
        self.project = equinox_dense(
            self.recurrent_size, self.recurrent_size, use_equinox_init=init
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LinearRNNRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LinearRNNRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        return jax.nn.leaky_relu(self.project(state))

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LinearRNNRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> LinearRNNRecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(LinearRNNSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
