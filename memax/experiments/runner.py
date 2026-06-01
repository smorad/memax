"""Training loops for equinox and linen experiment scripts."""

from typing import Any, Callable, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

from memax.experiments.config import ExperimentConfig
from memax.linen.train_utils import update_model as linen_update_model


def _num_batches(dataset_size: int, batch_size: int) -> int:
    return dataset_size // batch_size


def _update_indices(
    dataset_size: int, batch_size: int, max_updates: Optional[int]
) -> range:
    n = _num_batches(dataset_size, batch_size)
    if max_updates is not None:
        n = min(n, max_updates)
    return range(n)


def run_equinox_training(
    config: ExperimentConfig,
    name: str,
    model: eqx.Module,
    dataset: Dict[str, Any],
    loss_fn: Callable,
    wandb_module=None,
) -> Dict[str, float]:
    if config.use_wandb and wandb_module is not None:
        wandb_module.init(project=config.project_name, name=name)

    lr_schedule = optax.constant_schedule(config.lr)
    opt = optax.chain(optax.zero_nans(), optax.adamw(lr_schedule))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(config.seed)
    last_metrics: Dict[str, float] = {}

    for epoch in range(config.num_epochs):
        key, shuffle_key = jax.random.split(key)
        shuffle_idx = jax.random.permutation(shuffle_key, dataset["size"])
        x = dataset["x_train"][shuffle_idx]
        y = dataset["y_train"][shuffle_idx]
        updates = _update_indices(
            dataset["size"], config.batch_size, config.max_updates
        )
        pbar = tqdm.tqdm(updates, desc=f"{name} epoch {epoch}", leave=False)

        for update in pbar:
            key, subkey = jax.random.split(key)
            start = update * config.batch_size
            end = start + config.batch_size
            x_batch = x[start:end]
            y_batch = y[start:end]

            from memax.equinox.train_utils import update_model

            model, opt_state, metrics = eqx.filter_jit(update_model)(
                model=model,
                loss_fn=loss_fn,
                opt=opt,
                opt_state=opt_state,
                x=x_batch,
                y=y_batch,
                key=subkey,
            )
            last_metrics = {k: float(jnp.mean(v)) for k, v in metrics.items()}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in last_metrics.items()})
            if config.use_wandb and wandb_module is not None:
                wandb_module.log({**last_metrics, "epoch": epoch})

    if config.use_wandb and wandb_module is not None:
        wandb_module.finish()
    return last_metrics


def run_linen_training(
    config: ExperimentConfig,
    name: str,
    model,
    dataset: Dict[str, Any],
    loss_fn: Callable,
    wandb_module=None,
) -> Dict[str, float]:
    if config.use_wandb and wandb_module is not None:
        wandb_module.init(project=config.project_name, name=name)

    lr_schedule = optax.constant_schedule(config.lr)
    opt = optax.chain(optax.zero_nans(), optax.adamw(lr_schedule))
    key = jax.random.PRNGKey(config.seed)

    dummy_x = dataset["x_train"][0]
    dummy_starts = jnp.zeros(dummy_x.shape[0], dtype=bool)
    dummy_h = model.zero_carry()
    params = model.init(key, dummy_h, (dummy_x, dummy_starts))
    opt_state = opt.init(params)
    last_metrics: Dict[str, float] = {}

    jitted_update = jax.jit(linen_update_model, static_argnames=("loss_fn", "opt"))

    for epoch in range(config.num_epochs):
        key, shuffle_key = jax.random.split(key)
        shuffle_idx = jax.random.permutation(shuffle_key, dataset["size"])
        x = dataset["x_train"][shuffle_idx]
        y = dataset["y_train"][shuffle_idx]
        updates = _update_indices(
            dataset["size"], config.batch_size, config.max_updates
        )
        pbar = tqdm.tqdm(updates, desc=f"{name} epoch {epoch}", leave=False)

        for update in pbar:
            key, subkey = jax.random.split(key)
            start = update * config.batch_size
            end = start + config.batch_size
            x_batch = x[start:end]
            y_batch = y[start:end]

            params, opt_state, metrics = jitted_update(
                params, loss_fn, opt, opt_state, x_batch, y_batch, key=subkey
            )
            last_metrics = {k: float(jnp.mean(v)) for k, v in metrics.items()}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in last_metrics.items()})
            if config.use_wandb and wandb_module is not None:
                wandb_module.log({**last_metrics, "epoch": epoch})

    if config.use_wandb and wandb_module is not None:
        wandb_module.finish()
    return last_metrics
