import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train = tfds.load(name="mnist", split="train", data_dir="./mnist_data")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)
