"""This module contains training utilities for Equinox models.
It includes loss functions, accuracy metrics, and training loops.
It also provides a straightforward way to construct multi-layer recurrent models."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype.typing import Any, Callable, Dict, Optional, Tuple
from jaxtyping import Array, Shaped

from memax.equinox.groups import Module
from memax.equinox.models.multihead_residual import MultiHeadResidualModel
from memax.equinox.models.residual import ResidualModel
from memax.equinox.registry import (
    LAYER_REGISTRY,
    SEMIGROUP_REGISTRY,
    LayerFn,
    list_layers,
)


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


def mse(
    y_hat: Shaped[Array, "Batch ... Feature"], y: Shaped[Array, "Batch ... Feature"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.linalg.norm(y - y_hat, axis=-1, ord=2))


def l1_error(
    y_hat: Shaped[Array, "Batch ... Feature"], y: Shaped[Array, "Batch ... Feature"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.linalg.norm(y - y_hat, axis=-1, ord=1))


def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))


def loss_regress_terminal_output(
    model: Module,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Classes"],
    key=None,
) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
    """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
    return the mean square error loss between the true yn and predicted y1n.

    Args:
        model: Module
        x: (batch, time, in_feature)
        y: (batch, out_feature)

    Returns:
        loss: scalar
        info: dict
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    assert (
        x.shape[0] == y.shape[0]
    ), f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
    assert x.ndim == 3, f"expected 3d input, got {x.ndim}d"
    assert y.ndim == 2, f"expected 2d input, got {y.ndim}d"

    starts = jnp.zeros((batch_size, seq_len), dtype=bool)
    key, init_key, model_key = jax.random.split(key, 3)
    init_key = jax.random.split(init_key, batch_size)
    # TODO: These all initialize in the same state, probably do not want this
    h0 = eqx.filter_vmap(model.initialize_carry)(init_key)

    model_key = jax.random.split(model_key, batch_size)

    _, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
    # batch, time, feature
    y_pred = y_preds[:, -1]

    loss = mse(y_pred, y)
    l1 = l1_error(y_pred, y)
    return loss, {"loss": loss, "l1_error": l1}


def loss_classify_terminal_output(
    model: Module,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Classes"],
    key=None,
    decay=0.01,
) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
    """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
    return the cross entropy loss between the true yn and predicted y1n.

    Args:
        model: memax.groups.Module
        x: (batch, time, in_feature)
        y: (batch, num_classes)

    Returns:
        loss: scalar
        info: dict
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    assert (
        x.shape[0] == y.shape[0]
    ), f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
    assert x.ndim == 3, f"expected 3d input, got {x.ndim}d"
    assert y.ndim == 2, f"expected 2d input, got {y.ndim}d"

    starts = jnp.zeros((batch_size, seq_len), dtype=bool)
    key, init_key, model_key = jax.random.split(key, 3)
    init_key = jax.random.split(init_key, batch_size)
    # TODO: These all initialize in the same state, probably do not want this
    h0 = eqx.filter_vmap(model.initialize_carry)(init_key)

    model_key = jax.random.split(model_key, batch_size)

    h, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
    # batch, time, feature
    y_pred = y_preds[:, -1]
    loss = cross_entropy(y_pred, y)
    acc = accuracy(y_pred, y)
    return loss, {"loss": loss, "accuracy": acc}


def update_model(
    model: Module,
    loss_fn: Callable,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: Shaped[Array, "Batch ..."],
    y: Shaped[Array, "Batch ..."],
    key=None,
) -> Tuple[Module, optax.OptState, Dict[str, Array]]:
    """Update the model using the given loss function and optimizer."""
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y, key)
    updates, opt_state = opt.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_info


@eqx.filter_jit
def scan_one_epoch(
    model: Module,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable,
    xs: Shaped[Array, "Datapoint ..."],
    ys: Shaped[Array, "Datapoint ..."],
    batch_size: int,
    batch_index: Shaped[Array, "Batch ..."],
    *,
    key: jax.random.PRNGKey,
) -> Tuple[Module, optax.OptState, Dict[str, Array]]:
    """Train a single epoch using the scan operator. Functions as a dataloader and train loop."""
    assert (
        xs.shape[0] == ys.shape[0]
    ), f"batch size mismatch: {xs.shape[0]} != {ys.shape[0]}"
    params, static = eqx.partition(model, eqx.is_array)

    def get_batch(x, y, step):
        """Returns a specific batch of size `batch_size` from `x` and `y`."""
        start = step * batch_size
        x_batch = jax.lax.dynamic_slice_in_dim(x, start, batch_size, 0)
        y_batch = jax.lax.dynamic_slice_in_dim(y, start, batch_size, 0)
        return x_batch, y_batch

    def inner(carry, index):
        params, opt_state, key = carry
        x, y = get_batch(xs, ys, index)
        key = jax.random.split(key)[0]
        model = eqx.combine(params, static)
        # JIT this otherwise it takes ages to compile the epoch
        params, opt_state, metrics = update_model(
            model, loss_fn, opt, opt_state, x, y, key=key
        )
        params, _ = eqx.partition(params, eqx.is_array)
        return (params, opt_state, key), metrics

    (params, opt_state, key), epoch_metrics = jax.lax.scan(
        inner,
        (params, opt_state, key),
        batch_index,
    )
    model = eqx.combine(params, static)
    return model, opt_state, epoch_metrics


def get_semigroups(
    recurrent_size: int,
    semigroup_kwargs: Optional[Dict[str, Any]] = None,
    *,
    key: jax.random.PRNGKey,
) -> Dict[str, Module]:
    """Returns a dictionary containing all implemented semigroups.

    This returns the operator used in the scan, not the full recurrent cell.
    ``recurrent_size`` is the semigroup state dimension (for matrix cells, the
    matrix side length — see :mod:`memax.equinox.factory_docs`).
    """
    semigroup_kwargs = semigroup_kwargs or {}
    return {
        name: factory(
            recurrent_size,
            key=key,
            **semigroup_kwargs.get(name, {}),
        )
        for name, factory in SEMIGROUP_REGISTRY.items()
    }


def _bind_layer_kwargs(factory: LayerFn, overrides: Dict[str, Any]) -> LayerFn:
    if not overrides:
        return factory
    return lambda hidden_size, key, fn=factory, kw=overrides: fn(hidden_size, key, **kw)


def build_named_model(
    model_name: str,
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    *,
    key: jax.random.PRNGKey,
    layer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict] = None,
    num_heads: Optional[int] = None,
) -> Module:
    """Build one residual-trunk model for a registered cell name."""
    return build_model(
        input=input,
        hidden=hidden,
        output=output,
        num_layers=num_layers,
        models=[model_name],
        key=key,
        layer_kwargs=layer_kwargs,
        model_kwargs=model_kwargs,
        num_heads=num_heads,
    )[model_name]


def build_model(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    models: str = "all",
    *,
    key: jax.random.PRNGKey,
    layer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict] = None,
    num_heads: Optional[int] = None,
    extra_layers: Optional[Dict[str, LayerFn]] = None,
) -> Dict[str, Module]:
    """Build residual-trunk sequence models from registered memory cells.

    Each model stacks ``num_layers`` recurrent cells (semigroups or set actions) in a
    residual / DenseNet trunk with per-layer readout mixers. Pass ``num_heads`` to use
    :class:`~memax.equinox.models.multihead_residual.MultiHeadResidualModel`; omit it
    for :class:`~memax.equinox.models.residual.ResidualModel`.

    Returns a name-to-module mapping. With ``models="all"``, every registered cell is
    included; otherwise only the requested names are built.

    Per-cell ``layer_kwargs`` override factory defaults (e.g. ``recurrent_size``).
    See :mod:`memax.equinox.factory_docs` for sizing conventions — matrix-state
    cells use ``O(recurrent_size**2)`` memory, not ``O(hidden)``.
    """
    layer_kwargs = layer_kwargs or {}
    model_kwargs = model_kwargs or {}
    registry = {**LAYER_REGISTRY, **(extra_layers or {})}
    layers = {
        name: _bind_layer_kwargs(factory, layer_kwargs.get(name, {}))
        for name, factory in registry.items()
    }

    def make_trunk(make_layer_fn):
        trunk_kwargs = dict(
            make_layer_fn=make_layer_fn,
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=num_layers,
            key=key,
            **model_kwargs,
        )
        if num_heads is not None:
            return MultiHeadResidualModel(**trunk_kwargs, num_heads=num_heads)
        return ResidualModel(**trunk_kwargs)

    if models == "all":
        return {name: make_trunk(fn) for name, fn in layers.items()}

    missing = [name for name in models if name not in layers]
    if missing:
        available = ", ".join(sorted(list_layers()))
        raise KeyError(f"Unknown model(s): {missing}. Available layers: {available}")
    return {name: make_trunk(layers[name]) for name in models}
