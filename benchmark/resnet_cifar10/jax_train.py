import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import tensorflow_datasets as tfds
import time
import tensorflow as tf

# Define normalization constants
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]


# Define a normalization function
def process(image, label):
    # Random crop with padding
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)  # Add padding of 4 pixels
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0  # Convert to [0, 1]
    image = (image - mean) / std  # Normalize using mean and std
    return image, label


batch_size = 128
num_epochs = 30

# Load CIFAR-10 dataset
train_dataset = (
    tfds.load("cifar10", split="train", as_supervised=True)
    .cache()
    .map(process, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(50000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)


# Define LeNet architecture
class LeNet(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(6, kernel_size=(5, 5))(x)  # Output channels=6
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(16, kernel_size=(5, 5))(x)  # Output channels=16
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


# Initialize Model and Optimizer
rng = jax.random.PRNGKey(0)
model = LeNet()
params = model.init(rng, jnp.ones([1, 32, 32, 3]))
optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)
opt_state = optimizer.init(params)


# Define Loss and Update Function
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))


@jax.jit
def train_step(params, opt_state, images, labels):
    def loss_fn(params):
        logits = model.apply(params, images)
        return cross_entropy_loss(logits, labels)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Training Loop
for epoch in range(num_epochs):
    start_time = time.time()
    epoch_loss = 0
    for images, labels in train_dataset.as_numpy_iterator():
        images, labels = jnp.array(images), jnp.array(labels)
        params, opt_state, loss = train_step(params, opt_state, images, labels)
        epoch_loss += loss
    epoch_time = time.time() - start_time
    print(
        f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds, Loss: {epoch_loss/len(train_dataset):.4f}"
    )
