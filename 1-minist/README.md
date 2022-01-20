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
