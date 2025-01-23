import jax
from jax import numpy as jnp, random as jrandom
from jax import jit, vmap, grad, value_and_grad
from flax import linen as nn
import equinox as eqx

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from transformers import FlaxGPT2LMHeadModel


class LeNetFlaxNative(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv(features=16, kernel_size=(5, 5))
        self.dense1 = nn.Dense(features=120)
        self.dense2 = nn.Dense(features=84)
        self.dense3 = nn.Dense(features=10)

    # @eqx.filter_jit
    # @jit
    # def __call__(self, x: Float[Array, "b 32 32 3"]) -> Float[Array, "b 10"]:
    def __call__(self, x):
        # conv1
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # conv2
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # linear
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dense3(x)

        return x


class LeNetFlax(nn.Module):
    # @jit
    # @jaxtyped(typechecker)
    @nn.compact
    def __call__(self, x: Float[Array, "b 32 32 3"]) -> Float[Array, "b 10"]:
        # conv1
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # conv2
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # linear
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)

        return x


class LeNetEqx(eqx.Module):
    cnn: eqx.nn.Sequential

    def __init__(self, key: jrandom.PRNGKey):
        # keys = jrandom.split(key, 5)
        key1, key2, key3, key4, key5 = jrandom.split(key, 5)
        self.cnn = eqx.nn.Sequential([
            eqx.nn.Conv2d(3, 6, (5, 5), key=key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.MaxPool2d((2, 2), (2, 2)),
            eqx.nn.Conv2d(6, 16, (5, 5), key=key2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.MaxPool2d((2, 2), (2, 2)),
            eqx.nn.Lambda(jnp.ravel),
            eqx.nn.Linear(16 * 5 * 5, 120, key=key3),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(120, 84, key=key4),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(84, 10, key=key5)
        ])

    def __call__(self, x: Float[Array, "b 32 32 3"]) -> Float[Array, "b 10"]:
        return self.cnn(x)


if __name__ == '__main__':
    key = jrandom.PRNGKey(0)
    x = jrandom.normal(key, (20, 32, 32, 3))
    # x = jrandom.normal(key, (20, 3, 32, 32))
    model = LeNetFlaxNative()
    # model = LeNetFlax()
    # model = LeNetEqx(key)
    # # jax.tree_map(lambda x: print(type(x)), model)
    # print(model)
    print(model.tabulate(key, x))
    params = model.init(key, x)
    y = model.apply(params, x)
    print(y.shape)
    # print(vmap(model)(x))
    pass
