"""Test all models on a simple 'remember the first input in the sequence' task"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest

from memax.linen.models.residual import ResidualModel
from memax.linen.set_actions.elman import Elman
from memax.linen.set_actions.indrnn import IndRNN
from memax.linen.set_actions.lstm import LSTM
from memax.linen.set_actions.mgu import MGU
from memax.linen.set_actions.spherical import Spherical
from memax.linen.train_utils import get_residual_memory_models


def get_desired_accuracies_equinox():
    # Kept in sync with `tests/test_initial_input_equinox.py`.
    return {
        "Identity": 0,
        "Stack": 0,
        "Attention": 0.99,
        "Attention-RoPE": 0.99,
        "Attention-ALiBi": 0.99,
        "DLSE": 0.99,
        "FFM": 0.99,
        "FART": 0.99,
        "FWP": 0.99,
        "DeltaNet": 0.99,
        "DeltaProduct": 0.99,
        "GDN": 0.99,
        "TTTL": 0.99,
        "TTTL-RoPE": 0.99,
        "LRU": 0.99,
        "S6": 0.99,
        "LinearRNN": 0.99,
        "PSpherical": 0.99,
        "GRU": 0.99,
        "IndRNN": 0.99,
        "Elman": 0.60,
        "ElmanReLU": 0.60,
        "Spherical": 0.99,
        "NMax": 0.99,
        "MGU": 0.99,
        "LSTM": 0.99,
        "S6D": 0.99,
        "S6": 0.99,
    }


def get_desired_accuracies():
    eqx = get_desired_accuracies_equinox()
    return {k: eqx[k] for k in get_residual_memory_models(16, 3 - 1).keys()} | {
        "Elman": eqx["Elman"],
        "LSTM": eqx["LSTM"],
        "MGU": eqx["MGU"],
        "IndRNN": eqx["IndRNN"],
        "Spherical": eqx["Spherical"],
    }


def get_models(hidden: int, output: int, input: int = 3):
    model_kwargs = {"input": input}
    models = dict(
        get_residual_memory_models(hidden, output, model_kwargs=model_kwargs).items()
    )
    models.update(
        {
            "Elman": ResidualModel(
                make_layer_fn=lambda recurrent_size: Elman(
                    algebra=Elman.default_algebra(
                        recurrent_size=recurrent_size, activation=jax.nn.tanh
                    ),
                    scan=Elman.default_scan(),
                    recurrent_size=recurrent_size,
                    hidden_size=recurrent_size,
                ),
                recurrent_size=hidden,
                output_size=output,
                input=input,
            ),
            "LSTM": ResidualModel(
                make_layer_fn=lambda recurrent_size: LSTM(
                    algebra=LSTM.default_algebra(recurrent_size=recurrent_size),
                    scan=LSTM.default_scan(),
                    recurrent_size=recurrent_size,
                ),
                recurrent_size=hidden,
                output_size=output,
                input=input,
            ),
            "MGU": ResidualModel(
                make_layer_fn=lambda recurrent_size: MGU(
                    algebra=MGU.default_algebra(recurrent_size=recurrent_size),
                    scan=MGU.default_scan(),
                    recurrent_size=recurrent_size,
                ),
                recurrent_size=hidden,
                output_size=output,
                input=input,
            ),
            "IndRNN": ResidualModel(
                make_layer_fn=lambda recurrent_size: IndRNN(
                    algebra=IndRNN.default_algebra(
                        recurrent_size=recurrent_size,
                    ),
                    scan=IndRNN.default_scan(),
                    recurrent_size=recurrent_size,
                    hidden_size=recurrent_size,
                ),
                recurrent_size=hidden,
                output_size=output,
                input=input,
            ),
            "Spherical": ResidualModel(
                make_layer_fn=lambda recurrent_size: Spherical(
                    algebra=Spherical.default_algebra(recurrent_size=recurrent_size),
                    scan=Spherical.default_scan(),
                    recurrent_size=recurrent_size,
                    hidden_size=recurrent_size,
                ),
                recurrent_size=hidden,
                output_size=output,
                input=input,
            ),
        }
    )
    return models


def ce_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


@pytest.mark.parametrize(
    "model_name, model",
    get_models(16, 3 - 1).items(),
)
def test_initial_input(
    model_name,
    model,
    epochs=400,
    num_seqs=5,
    seq_len=20,
    input_dims=3,
):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    opt = optax.adam(learning_rate=3e-3)

    # init model
    key = jax.random.PRNGKey(0)
    dummy_x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
    dummy_x = jax.nn.one_hot(dummy_x, input_dims - 1)
    dummy_x = jnp.concatenate(
        [dummy_x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1
    )
    dummy_h = model.zero_carry()
    dummy_starts = jnp.zeros(dummy_x.shape[0], dtype=bool)
    params = model.init(key, dummy_h, (dummy_x, dummy_starts))
    init_carry_fn = partial(model.apply, method="initialize_carry")
    apply_fn = model.apply
    state = opt.init(params)

    def error(params, key):
        h = init_carry_fn(params)
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

        _, y_hat = apply_fn(params, h, (x, start))
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    loss_fn = jax.jit(jax.grad(error, has_aux=True))
    losses = []
    accuracies = []
    for epoch in range(epochs):
        key, _ = jax.random.split(key)
        grads, loss_info = loss_fn(params, key)
        updates, state = jax.jit(opt.update)(grads, state)
        params = optax.apply_updates(params, updates)
        losses.append(loss_info["loss"])
        accuracies.append(loss_info["accuracy"])

    losses, accuracies = jnp.stack(losses), jnp.stack(accuracies)
    losses = losses[-100:].mean()
    accuracies = accuracies[-100:].mean()
    print(f"{model_name} mean accuracy: {accuracies:0.3f}")
    assert (
        accuracies >= get_desired_accuracies()[model_name]
    ), f"Failed {model_name}, expected {get_desired_accuracies()[model_name]}, got {accuracies}"

    # Verify recurrent mode works well too
    def rerror(params, key):
        h = init_carry_fn(params)
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)
        y_hats = []

        for t in range(timesteps):
            h, y_hat = apply_fn(params, h, (x[t : t + 1], start[t : t + 1]))
            h = model.latest_recurrent_state(h)
            y_hats.append(y_hat)

        y_hat = jnp.concatenate(y_hats, axis=0)
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    _, r_metrics = rerror(params, key)
    assert (
        r_metrics["accuracy"] >= get_desired_accuracies()[model_name]
    ), f"Failed {model_name} (recurrent mode), expected {get_desired_accuracies()[model_name]}, got {r_metrics['accuracy']}"


if __name__ == "__main__":
    test_initial_input()
