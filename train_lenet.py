import tensorflow as tf
import tensorflow_datasets as tfds

# Define normalization constants
mean = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784]


# Define a normalization function
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Convert to [0, 1]
    image = (image - mean) / std  # Normalize using mean and std
    return image, label


# Define a function for data augmentation
def augment_image(image, label):
    # Random crop with padding
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)  # Add padding of 4 pixels
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    return image, label


# Load CIFAR-10 dataset
train_dataset = tfds.load("cifar10", split="train", as_supervised=True)

# Apply transformations: augment and normalize
train_dataset = (
    train_dataset.map(augment_image)
    .map(normalize_image)
    .shuffle(10000)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

i = 0
for i, (image, label) in enumerate(train_dataset.as_numpy_iterator()):
    print(i, type(image), image.shape, type(label), label.shape)
