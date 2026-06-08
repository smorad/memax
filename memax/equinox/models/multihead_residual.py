import jax
from beartype.typing import List, Optional, Tuple
from equinox import filter_vmap, nn
from jaxtyping import PRNGKeyArray, Shaped

from memax.equinox.groups import Module
from memax.equinox.models.layer_mixer import LayerMixer
from memax.mtypes import Input, ResetRecurrentState


class MultiHeadResidualModel(Module):
    """A residual stack where per-layer ``ff`` is replaced by :class:`LayerMixer`."""

    layers: List[Module]
    mixers: List[LayerMixer]
    map_in: nn.Linear
    map_out: nn.Linear
    recurrent_size: int
    num_heads: int

    def __init__(
        self,
        make_layer_fn,
        input_size,
        output_size,
        recurrent_size,
        num_heads,
        num_layers=2,
        activation=jax.nn.leaky_relu,
        *,
        key,
    ):
        if recurrent_size % num_heads != 0:
            raise ValueError(
                f"recurrent_size ({recurrent_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.recurrent_size = recurrent_size
        self.num_heads = num_heads
        self.layers = []
        self.mixers = []
        keys = jax.random.split(key, 3)
        self.map_in = nn.Linear(input_size, recurrent_size, key=keys[0])
        self.map_out = nn.Linear(recurrent_size, output_size, key=keys[1])
        key = keys[2]
        for _ in range(num_layers):
            key, layer_key, mixer_key = jax.random.split(key, 3)
            layer = make_layer_fn(recurrent_size=recurrent_size, key=layer_key)
            self.layers.append(layer)
            self.mixers.append(
                LayerMixer(
                    layer.readout_dim,
                    recurrent_size,
                    num_heads=num_heads,
                    activation=activation,
                    key=mixer_key,
                )
            )

    def __call__(
        self, h: ResetRecurrentState, x: Input, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for mixer, recurrent_layer, h_i in zip(self.mixers, self.layers, h):
            if key is None:
                key, rkey = None, None
            else:
                key, rkey = jax.random.split(key)
            tmp, feat = recurrent_layer(h_i, layer_in, key=rkey)
            h_out.append(tmp)
            z = filter_vmap(mixer)(feat)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        if key is None:
            keys = tuple(None for _ in range(len(self.layers)))
        else:
            keys = jax.random.split(key, len(self.layers))
        return tuple(l.initialize_carry(k) for l, k in zip(self.layers, keys))

    def latest_recurrent_state(self, h: ResetRecurrentState) -> ResetRecurrentState:
        return tuple(l.latest_recurrent_state(h_i) for l, h_i in zip(self.layers, h))
