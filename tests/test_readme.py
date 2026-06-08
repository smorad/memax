def test_readme():
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from memax.equinox.semigroups.lru import LRU, LRUSemigroup, make_layer

    key = jax.random.key(0)
    hidden_size = 16

    layer = make_layer(hidden_size, key)
    layer = LRU(hidden_size=hidden_size, recurrent_size=hidden_size, key=key)
    sg = LRUSemigroup(recurrent_size=hidden_size)

    sequence_starts = jnp.array([True, False, False, True, False])
    x = jnp.zeros((5, hidden_size))
    inputs = (x, sequence_starts)

    h = eqx.filter_jit(layer.initialize_carry)()
    h, y = eqx.filter_jit(layer)(h, inputs)
    assert y.shape == (5, hidden_size)


def test_readme_quickstart():
    import jax
    import jax.numpy as jnp
    from equinox import filter_jit, filter_vmap

    from memax.equinox.train_utils import add_batch_dim, build_named_model

    T, F = 5, 6  # time and feature dim

    model = build_named_model(
        model_name="LRU",
        input=F,
        hidden=8,
        output=1,
        num_layers=2,
        key=jax.random.key(0),
    )

    starts = jnp.array([True, False, False, True, False])
    xs = jnp.zeros((T, F))
    hs, ys = filter_jit(model)(model.initialize_carry(), (xs, starts))
    last_h = filter_jit(model.latest_recurrent_state)(hs)

    # with batch dim
    B = 4
    starts = jnp.zeros((B, T), dtype=bool)
    xs = jnp.zeros((B, T, F))
    hs_0 = add_batch_dim(model.initialize_carry(), B)
    hs, ys = filter_jit(filter_vmap(model))(hs_0, (xs, starts))


if __name__ == "__main__":
    test_readme()
    test_readme_quickstart()
