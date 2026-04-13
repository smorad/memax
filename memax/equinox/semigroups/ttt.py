from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

TTTRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
TTTRecurrentStateWithReset = Tuple[TTTRecurrentState, StartFlag]


class TTTLinearSemigroup(Semigroup):
    """The associative operation for the TTT-Linear gradient descent step."""

    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentState:
        # Initial hidden state is an identity matrix (M) and zero matrix (X)
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size))
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        carry: TTTRecurrentState,
        input: TTTRecurrentState,
    ) -> TTTRecurrentState:
        # Exact same associative factorization as GDN: S_t = M_t S_{t-1} + X_t
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class TTTLinear(GRAS):
    """The minimalist Test-Time Training (Linear) architecture.
    
    Replaces data-dependent gating with an inner-loop gradient descent step (eta) 
    to minimize a self-supervised reconstruction loss.
    """

    hidden_size: int
    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [TTTRecurrentStateWithReset, TTTRecurrentStateWithReset],
                TTTRecurrentStateWithReset,
            ],
            TTTRecurrentStateWithReset,
            TTTRecurrentStateWithReset,
        ],
        TTTRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    output: nn.Linear
    eta: Float[Array, ""]  # The learned inner-loop learning rate

    def __init__(self, hidden_size: int, recurrent_size: int, key: PRNGKeyArray):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(TTTLinearSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 4)

        self.K = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[2])
        self.output = nn.Linear(recurrent_size, hidden_size, key=keys[3])
        
        # Initialize the inner learning rate as a learnable parameter.
        self.eta = jnp.array(0.1)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentStateWithReset:
        emb, start = x
        
        k = self.K(emb)
        # Normalization is critical in TTT to prevent explosive state growth
        # during the unrolled gradient descent steps.
        k = k / (jnp.linalg.norm(k) + 1e-6) 
        
        v = self.V(emb)

        # The TTT-Linear gradient update mapped to the semigroup components
        M = jnp.eye(self.recurrent_size) - self.eta * jnp.outer(k, k)
        X = self.eta * jnp.outer(v, k)
        
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: TTTRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (M, X), reset_flag = h
        
        q = self.Q(emb)
        q = q / (jnp.linalg.norm(q) + 1e-6) 
        
        return self.output(X @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> TTTRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)