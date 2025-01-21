import jax
from jax import numpy as jnp, random as jrandom
from flax import linen as nn


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)

        x = nn.Dense(features=84)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


if __name__ == '__main__':
    key = jrandom.PRNGKey(0)
    x = jrandom.normal(key, (1, 28, 28, 1))
    model = LeNet()
    params = model.init(key, x)['params']
    y = model.apply({'params': params}, x)
    print(y.shape)
