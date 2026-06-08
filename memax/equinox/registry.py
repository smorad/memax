"""Name-to-factory registries for equinox memory cells.

See :mod:`memax.equinox.factory_docs` for how ``hidden_size`` and
``recurrent_size`` are interpreted in ``make_layer`` / ``make_semigroup``.

Variant registry names (e.g. ``Attention-RoPE``) bind default kwargs on the
module's single ``make_layer`` / ``make_semigroup`` via :func:`functools.partial`.
"""

from functools import partial

import jax
from beartype.typing import Callable

from memax.equinox.groups import Module
from memax.equinox.semigroups.attn import make_layer as attention_layer
from memax.equinox.semigroups.attn import make_semigroup as attention_semigroup
from memax.equinox.semigroups.delta import make_layer as deltanet_layer
from memax.equinox.semigroups.delta import make_semigroup as deltanet_semigroup
from memax.equinox.semigroups.deltap import make_layer as deltap_layer
from memax.equinox.semigroups.deltap import make_semigroup as deltap_semigroup
from memax.equinox.semigroups.fart import make_layer as fart_layer
from memax.equinox.semigroups.fart import make_semigroup as fart_semigroup
from memax.equinox.semigroups.ffm import make_layer as ffm_layer
from memax.equinox.semigroups.ffm import make_semigroup as ffm_semigroup
from memax.equinox.semigroups.fwp import make_layer as fwp_layer
from memax.equinox.semigroups.fwp import make_semigroup as fwp_semigroup
from memax.equinox.semigroups.gdn import make_layer as gdn_layer
from memax.equinox.semigroups.gdn import make_semigroup as gdn_semigroup
from memax.equinox.semigroups.identity import make_layer as identity_layer
from memax.equinox.semigroups.lrnn import make_layer as linearrnn_layer
from memax.equinox.semigroups.lrnn import make_semigroup as linearrnn_semigroup
from memax.equinox.semigroups.lru import make_layer as lru_layer
from memax.equinox.semigroups.lru import make_semigroup as lru_semigroup
from memax.equinox.semigroups.nmax import make_layer as nmax_layer
from memax.equinox.semigroups.nmax import make_semigroup as nmax_semigroup
from memax.equinox.semigroups.s6 import make_layer as s6_layer
from memax.equinox.semigroups.s6 import make_semigroup as s6_semigroup
from memax.equinox.semigroups.spherical import make_layer as pspherical_layer
from memax.equinox.semigroups.spherical import make_semigroup as pspherical_semigroup
from memax.equinox.semigroups.stack import make_layer as stack_layer
from memax.equinox.semigroups.stack import make_semigroup as stack_semigroup
from memax.equinox.semigroups.ttt import make_layer as tttl_layer
from memax.equinox.semigroups.ttt import make_semigroup as tttl_semigroup
from memax.equinox.set_actions.elman import make_layer as elman_layer
from memax.equinox.set_actions.gru import make_layer as gru_layer
from memax.equinox.set_actions.indrnn import make_layer as indrnn_layer
from memax.equinox.set_actions.lstm import make_layer as lstm_layer
from memax.equinox.set_actions.mgu import make_layer as mgu_layer
from memax.equinox.set_actions.spherical import make_layer as spherical_layer

LayerFn = Callable[[int, jax.Array], Module]  # (hidden_size, key) -> GRAS
SemigroupFn = Callable[..., Module]

LAYER_REGISTRY: dict[str, LayerFn] = {
    "Identity": identity_layer,
    "NMax": nmax_layer,
    "FART": fart_layer,
    "FWP": fwp_layer,
    "DeltaNet": deltanet_layer,
    "DeltaProduct": deltap_layer,
    "GDN": gdn_layer,
    "TTTL": tttl_layer,
    "TTTL-RoPE": partial(tttl_layer, positional_embedding="rope"),
    "FFM": ffm_layer,
    "S6": s6_layer,
    "PSpherical": pspherical_layer,
    "LRU": lru_layer,
    "LinearRNN": linearrnn_layer,
    "Stack": stack_layer,
    "Attention": attention_layer,
    "Attention-RoPE": partial(attention_layer, positional_embedding="rope"),
    "Attention-ALiBi": partial(attention_layer, positional_embedding="alibi"),
    "GRU": gru_layer,
    "Elman": elman_layer,
    "ElmanReLU": partial(elman_layer, activation=jax.nn.relu),
    "IndRNN": indrnn_layer,
    "Spherical": spherical_layer,
    "MGU": mgu_layer,
    "LSTM": lstm_layer,
}

SEMIGROUP_REGISTRY: dict[str, SemigroupFn] = {
    "PSpherical": pspherical_semigroup,
    "FFM": ffm_semigroup,
    "FART": fart_semigroup,
    "LinearRNN": linearrnn_semigroup,
    "LRU": lru_semigroup,
    "S6": s6_semigroup,
    "NMax": nmax_semigroup,
    "FWP": fwp_semigroup,
    "DeltaNet": deltanet_semigroup,
    "DeltaProduct": deltap_semigroup,
    "TTTL": tttl_semigroup,
    "TTTL-RoPE": partial(tttl_semigroup, use_rope=True),
    "GDN": gdn_semigroup,
    "Stack": stack_semigroup,
    "Attention": attention_semigroup,
}


def list_layers() -> tuple[str, ...]:
    return tuple(LAYER_REGISTRY)


def list_semigroups() -> tuple[str, ...]:
    return tuple(SEMIGROUP_REGISTRY)
