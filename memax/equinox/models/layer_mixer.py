import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Callable, Tuple
from equinox import nn
from jaxtyping import Array, PRNGKeyArray, Shaped


class LayerMixer(eqx.Module):
    """Per-head linears from readout features to trunk width, then norm and activation.

    Applies ``H`` independent ``Linear(in_features, head_dim)`` maps, concatenates to
    ``out_features``, then LayerNorm and activation.
    """

    num_heads: int
    head_dim: int
    in_features: int
    out_features: int
    heads: Tuple[nn.Linear, ...]
    norm: nn.LayerNorm
    activation: eqx.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        *,
        key: Shaped[PRNGKeyArray, ""],
    ):
        if out_features % num_heads != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        keys = jax.random.split(key, num_heads)
        self.heads = tuple(
            nn.Linear(in_features, self.head_dim, key=head_key) for head_key in keys
        )
        self.norm = nn.LayerNorm((out_features,), use_weight=False, use_bias=False)
        self.activation = nn.Lambda(activation)

    def __call__(self, z: Array) -> Array:
        ys = jnp.stack([head(z) for head in self.heads])
        y = ys.reshape(-1)
        y = self.norm(y)
        return self.activation(y)
