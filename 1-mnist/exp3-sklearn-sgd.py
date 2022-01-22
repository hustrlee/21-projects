import time
from datetime import timedelta
from joblib import parallel_backend

import tensorflow_datasets as tfds

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

mnist_train, mnist_info = tfds.load(name="mnist",
                                    split="train",
                                    data_dir="./mnist_data/",
                                    with_info=True)
mnist_test = tfds.load(name="mnist", split="test", data_dir="./mnist_data/")

# 将 tf.data.Dataset 格式转换成 pd.dataframe 格式
mnist_train = tfds.as_dataframe(mnist_train, mnist_info)
mnist_test = tfds.as_dataframe(mnist_test, mnist_info)

# 将 mnist 数据拍平
mnist_train["image"] = mnist_train["image"].map(lambda x: x.reshape(-1))
mnist_test["image"] = mnist_test["image"].map(lambda x: x.reshape(-1))

# 将 pd.dataframe 格式数据分解成 X_train, Y_train, x_test, y_test
X_train, y_train = mnist_train["image"], mnist_train["label"]
X_test, y_test = mnist_test["image"], mnist_test["label"]

# 将数据格式转换为 list
X_train, X_test, y_train, y_test = list(X_train), list(X_test), list(y_train), list(y_test)  # noqa

# 使用 pipeline
est = make_pipeline(StandardScaler(), SGDClassifier())

print("训练开始：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
start = time.time()

with parallel_backend("threading", n_jobs=10):
    est.fit(X_train, y_train)

delta = (time.time() - start)
elapsed = str(timedelta(seconds=delta))

print("训练耗时：", format(elapsed))

y_predict = est.predict(X_test)
print("acc:", est.score(X_test, y_test))

print(classification_report(y_test, y_predict))
