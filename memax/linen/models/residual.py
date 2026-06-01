import flax.linen as nn
import jax
from beartype.typing import Callable, Optional, Tuple
from jaxtyping import PRNGKeyArray, Shaped

from memax.linen.groups import Module
from memax.linen.inits import dense as equinox_dense
from memax.mtypes import Input, ResetRecurrentState


class ResidualModel(Module):
    """A model consisting of multiple recurrent layers, with a
    residual connection from the original input into each layer.

    There is a nonlinearity between network layers."""

    make_layer_fn: Callable[..., nn.Module]
    output_size: int
    recurrent_size: int
    input: Optional[int] = None
    num_layers: int = 2
    activation: Callable[[jax.Array], jax.Array] = jax.nn.leaky_relu
    use_equinox_init: bool = True

    def setup(self):
        layers = []
        ff = []
        init = self.use_equinox_init
        r, o = self.recurrent_size, self.output_size
        if init and self.input is not None:
            self.map_in = equinox_dense(r, self.input, use_equinox_init=True)
        else:
            self.map_in = nn.Dense(r)
        self.map_out = equinox_dense(o, r, use_equinox_init=init)
        for _ in range(self.num_layers):
            layers.append(self.make_layer_fn(recurrent_size=self.recurrent_size))
            ff.append(
                nn.Sequential(
                    [
                        equinox_dense(r, r, use_equinox_init=init),
                        nn.LayerNorm(use_scale=False, use_bias=False),
                        self.activation,
                    ]
                )
            )
        self.layers = layers
        self.ff = ff

    def __call__(
        self, h: ResetRecurrentState, x: Input
    ) -> Tuple[ResetRecurrentState, ...]:
        emb, start = x
        emb = jax.vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, recurrent_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = recurrent_layer(h_i, layer_in)
            h_out.append(tmp)
            z = jax.vmap(ff)(z)
            layer_in = (z, start)
        out = jax.vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        if key is None:
            keys = tuple(None for _ in range(self.num_layers))
        else:
            keys = jax.random.split(key, self.num_layers)

        return tuple(l.initialize_carry(k) for l, k in zip(self.layers, keys))

    @nn.nowrap
    def zero_carry(self) -> Tuple[ResetRecurrentState, ...]:
        layers = [
            self.make_layer_fn(recurrent_size=self.recurrent_size)
            for _ in range(self.num_layers)
        ]
        return tuple(l.zero_carry() for l in layers)

    @nn.nowrap
    def latest_recurrent_state(self, h: ResetRecurrentState) -> ResetRecurrentState:
        layers = [
            self.make_layer_fn(recurrent_size=self.recurrent_size)
            for _ in range(self.num_layers)
        ]
        return tuple(l.latest_recurrent_state(h_i) for l, h_i in zip(layers, h))
