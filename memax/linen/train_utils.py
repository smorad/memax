"""This module contains training utilities for Flax Linen models.
It includes loss functions, accuracy metrics, and training loops.
It also provides a straightforward way to construct multi-layer recurrent models."""

import math

import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any, Callable, Dict, Optional, Tuple
from flax.core import FrozenDict
from jaxtyping import Array, Shaped

from memax.linen.models.residual import ResidualModel
from memax.linen.semigroups.attn import Attention, AttentionSemigroup
from memax.linen.semigroups.delta import DeltaNet, DeltaNetSemigroup
from memax.linen.semigroups.deltap import DeltaProduct, DeltaProductSemigroup
from memax.linen.semigroups.fart import FART, FARTSemigroup
from memax.linen.semigroups.ffm import FFM, FFMSemigroup
from memax.linen.semigroups.fwp import FWP, FWPSemigroup
from memax.linen.semigroups.gdn import GDN, GDNSemigroup
from memax.linen.semigroups.lrnn import LinearRecurrent, LinearRNNSemigroup
from memax.linen.semigroups.lru import LRU, LRUSemigroup
from memax.linen.semigroups.s6 import S6, S6Semigroup
from memax.linen.semigroups.spherical import PSpherical, PSphericalSemigroup
from memax.linen.semigroups.stack import Stack, StackSemigroup
from memax.linen.semigroups.ttt import TTTLinear, TTTLinearSemigroup
from memax.linen.set_actions.gru import GRU


def add_batch_dim(h, batch_size: int, axis: int = 0) -> Shaped[Array, "Batch ..."]:
    """Given an recurrent state (pytree) `h`, add a new batch dimension of size `batch_size`.

    E.g., add_batch_dim(h, 32) will return a new state with shape (32, *h.shape). The state will
    be repeated along the new batch dimension.
    """
    expand = lambda x: jnp.repeat(jnp.expand_dims(x, axis), batch_size, axis=axis)
    h = jax.tree.map(expand, h)
    return h


def cross_entropy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))


def update_model(
    params: FrozenDict,
    loss_fn: Callable,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: Shaped[Array, "Batch ..."],
    y: Shaped[Array, "Batch ..."],
    key=None,
) -> Tuple[FrozenDict, optax.OptState, Dict[str, Array]]:
    grads, loss_info = jax.grad(loss_fn, has_aux=True)(params, x, y)
    updates, opt_state = opt.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_info


def loss_classify_terminal_output(
    params: FrozenDict,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Classes"],
    init_carry_fn,
    model_apply_fn,
) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    starts = jnp.zeros((batch_size, seq_len), dtype=bool)
    h0 = init_carry_fn(params)
    # h0 = jax.tree_map(partial(add_batch_dim, batch_size=batch_size), h0)
    h0 = add_batch_dim(h0, batch_size)

    _, y_preds = jax.vmap(model_apply_fn, in_axes=[None, 0, 0])(params, h0, (x, starts))
    y_pred = y_preds[:, -1]

    loss = cross_entropy(y_pred, y)
    acc = accuracy(y_pred, y)
    return loss, {"loss": loss, "accuracy": acc}


def get_semigroups(
    recurrent_size: int,
    semigroup_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, FrozenDict]:
    """Returns a dictionary containing all implemented semigroups.

    This returns the operator used in the scan, not the full recurrent cell.
    """
    semigroup_kwargs = semigroup_kwargs or {}
    return {
        "PSpherical": PSphericalSemigroup(
            recurrent_size, **semigroup_kwargs.get("PSpherical", {})
        ),
        "FFM": FFMSemigroup(
            trace_size=recurrent_size,
            context_size=recurrent_size,
            **semigroup_kwargs.get("FFM", {}),
        ),
        "FART": FARTSemigroup(recurrent_size, **semigroup_kwargs.get("FART", {})),
        "LinearRNN": LinearRNNSemigroup(
            recurrent_size, **semigroup_kwargs.get("LinearRNN", {})
        ),
        "LRU": LRUSemigroup(recurrent_size, **semigroup_kwargs.get("LRU", {})),
        "S6": S6Semigroup(recurrent_size, **semigroup_kwargs.get("S6", {})),
        "FWP": FWPSemigroup(recurrent_size, **semigroup_kwargs.get("FWP", {})),
        "DeltaNet": DeltaNetSemigroup(
            recurrent_size, **semigroup_kwargs.get("DeltaNet", {})
        ),
        "DeltaProduct": DeltaProductSemigroup(
            recurrent_size, **semigroup_kwargs.get("DeltaProduct", {})
        ),
        "TTTL": TTTLinearSemigroup(recurrent_size, **semigroup_kwargs.get("TTTL", {})),
        "TTTL-RoPE": TTTLinearSemigroup(
            recurrent_size,
            **semigroup_kwargs.get("TTTL", {"use_rope": True}),
        ),
        "GDN": GDNSemigroup(recurrent_size, **semigroup_kwargs.get("GDN", {})),
        "Stack": StackSemigroup(
            recurrent_size=recurrent_size,
            **semigroup_kwargs.get("Stack", {"stack_size": 4}),
        ),
        "Attention": AttentionSemigroup(
            recurrent_size=recurrent_size,
            **semigroup_kwargs.get("Attention", {"window_size": 4}),
        ),
    }


def get_residual_memory_models(
    hidden: int,
    output: int,
    num_layers: int = 2,
    models: str = "all",
    input: Optional[int] = None,
    layer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict] = None,
) -> Dict:
    """Constructs a trunk of stacked memory cells."""
    layer_kwargs = layer_kwargs or {}
    model_kwargs = dict(model_kwargs or {})
    if input is not None:
        model_kwargs.setdefault("input", input)
    layers = {
        "FART": lambda recurrent_size: FART(
            algebra=FART.default_algebra(recurrent_size=round(recurrent_size**0.5)),
            scan=FART.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            **layer_kwargs.get("FART", {}),
        ),
        "FWP": lambda recurrent_size: FWP(
            algebra=FWP.default_algebra(recurrent_size=round(recurrent_size**0.5)),
            scan=FWP.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            **layer_kwargs.get("FWP", {}),
        ),
        "DeltaNet": lambda recurrent_size: DeltaNet(
            algebra=DeltaNet.default_algebra(recurrent_size=round(recurrent_size**0.5)),
            scan=DeltaNet.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            **layer_kwargs.get("DeltaNet", {}),
        ),
        "DeltaProduct": lambda recurrent_size: DeltaProduct(
            algebra=DeltaProduct.default_algebra(
                recurrent_size=round(recurrent_size**0.5)
            ),
            scan=DeltaProduct.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            rank=4,
            **layer_kwargs.get("DeltaProduct", {}),
        ),
        "GDN": lambda recurrent_size: GDN(
            algebra=GDN.default_algebra(recurrent_size=round(recurrent_size**0.5)),
            scan=GDN.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            **layer_kwargs.get("GDN", {}),
        ),
        # TODO: Re-enable once TTTL perf matches equinox equivalent
        # "TTTL": lambda recurrent_size: TTTLinear(
        #     algebra=TTTLinear.default_algebra(
        #         recurrent_size=round(recurrent_size**0.5)
        #     ),
        #     scan=TTTLinear.default_scan(),
        #     hidden_size=recurrent_size,
        #     recurrent_size=round(recurrent_size**0.5),
        #     **layer_kwargs.get("TTTL", {}),
        # ),
        "TTTL-RoPE": lambda recurrent_size: TTTLinear(
            algebra=TTTLinear.default_algebra(
                recurrent_size=math.ceil(recurrent_size**0.5 / 2) * 2,
                use_rope=True,
            ),
            scan=TTTLinear.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=math.ceil(recurrent_size**0.5 / 2) * 2,
            positional_embedding="rope",
            **layer_kwargs.get("TTTL-RoPE", {}),
        ),
        "FFM": lambda recurrent_size: FFM(
            algebra=FFM.default_algebra(
                trace_size=4,
                context_size=recurrent_size // 4,
            ),
            scan=FFM.default_scan(),
            hidden_size=recurrent_size,
            trace_size=4,
            context_size=recurrent_size // 4,
            **layer_kwargs.get("FFM", {}),
        ),
        "S6": lambda recurrent_size: S6(
            algebra=S6.default_algebra(recurrent_size=recurrent_size),
            scan=S6.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            **layer_kwargs.get("S6", {}),
        ),
        "PSpherical": lambda recurrent_size: PSpherical(
            algebra=PSpherical.default_algebra(
                recurrent_size=round(recurrent_size**0.5)
            ),
            scan=PSpherical.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            **layer_kwargs.get("PSpherical", {}),
        ),
        "LRU": lambda recurrent_size: LRU(
            algebra=LRU.default_algebra(recurrent_size=recurrent_size),
            scan=LRU.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            **layer_kwargs.get("LRU", {}),
        ),
        "LinearRNN": lambda recurrent_size: LinearRecurrent(
            algebra=LinearRecurrent.default_algebra(recurrent_size=recurrent_size),
            scan=LinearRecurrent.default_scan(),
            recurrent_size=recurrent_size,
            **layer_kwargs.get("LinearRNN", {}),
        ),
        "Stack": lambda recurrent_size: Stack(
            algebra=Stack.default_algebra(recurrent_size=recurrent_size, window_size=4),
            scan=Stack.default_scan(),
            recurrent_size=recurrent_size,
            stack_size=4,
            **layer_kwargs.get("Stack", {}),
        ),
        "Attention": lambda recurrent_size: Attention(
            algebra=Attention.default_algebra(
                recurrent_size=recurrent_size, window_size=20
            ),
            scan=Attention.default_scan(),
            recurrent_size=recurrent_size,
            window_size=20,
            positional_embedding=None,
            **layer_kwargs.get("Attention", {}),
        ),
        "Attention-RoPE": lambda recurrent_size: Attention(
            algebra=Attention.default_algebra(
                recurrent_size=recurrent_size, window_size=20
            ),
            scan=Attention.default_scan(),
            recurrent_size=recurrent_size,
            window_size=20,
            positional_embedding="rope",
            **layer_kwargs.get("Attention-RoPE", {}),
        ),
        "Attention-ALiBi": lambda recurrent_size: Attention(
            algebra=Attention.default_algebra(
                recurrent_size=recurrent_size, window_size=20
            ),
            scan=Attention.default_scan(),
            recurrent_size=recurrent_size,
            window_size=20,
            positional_embedding="alibi",
            **layer_kwargs.get("Attention-ALiBi", {}),
        ),
        "GRU": lambda recurrent_size: GRU(
            algebra=GRU.default_algebra(recurrent_size=recurrent_size),
            scan=GRU.default_scan(),
            recurrent_size=recurrent_size,
            **layer_kwargs.get("GRU", {}),
        ),
    }
    if models == "all":
        return {
            name: ResidualModel(
                make_layer_fn=fn,
                recurrent_size=hidden,
                output_size=output,
                num_layers=num_layers,
                **model_kwargs,
            )
            for name, fn in layers.items()
        }
    else:
        return {
            name: ResidualModel(
                make_layer_fn=layers[name],
                recurrent_size=hidden,
                output_size=output,
                num_layers=num_layers,
                **model_kwargs,
            )
            for name in models
        }
