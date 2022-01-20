import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train, info = tfds.load(
    name="mnist", split="train", data_dir="./mnist_data", with_info=True)
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)

fig = tfds.show_examples(info, mnist_train)
