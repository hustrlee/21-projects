# MNIST 机器学习入门



## MNIST 数据集

Tensorflow Datasets 提供了一系列可以和 Tensorflow 配合使用的数据集。它负责下载和准备数据，以及构建 `tf.data.Dataset`。

### 安装 Tensorflow Datasets

```
pip install tensorflow-datasets
```

### 引用 Tensorflow Datasets

```python
import tensorflow_datasets as tfds
```

### 列出可用的数据集

```python
tfds.list_builders()
```

> - Tensorflow 官方在不断补充 Datasets 中的数据集，例如：最近添加了 `covid19` 新冠数据集。
> - `tensorflow_datasets` 需从 Google 获取数据，因此需要科学上网。

### `tfds.load()`：一行代码获取数据集

`tfds.load()` 是构建并加载 `tf.data.Dataset` 最简单的方式。

> `tf.data.Datase` 是构建输入流水线的标准 Tensorflow 接口。

- `tads.load()` 将下载并准备好数据集，一旦数据集下载成功，后续的 `load` 命令不会重新下载，可以重复使用准备好的数据。

- 可以指定 `download=False` 来阻止立即下载数据集。
- 可以指定 `data_dir=` 来自定义数据保存/加载的路径，默认是：`~/tensorflow_datasets/`。

```python
mnist_train = tfds.load(name="mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)
```

### 特征字典

所有 [`tfds`](https://www.tensorflow.org/datasets/api_docs/python/tfds?hl=zh-cn) 数据集都包含将特征名称映射到 Tensor 值的特征字典。 典型的数据集（如 MNIST）将具有2个键：`"image"` 和 `"label"`。 下面我们看一个例子。

> 注意：在图模式（graph mode）下，请参阅 [tf.data 指南](https://www.tensorflow.org/guide/data) 以了解如何在 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=zh-cn) 上进行迭代。
