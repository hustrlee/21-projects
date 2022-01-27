import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Model Configuration
dataset_shape = x_train.shape
input_shape = (dataset_shape[1] * dataset_shape[2], )
optimizer = Adam()

# Model
model = Sequential()
model.add(Flatten())
model.add(Dense(units=64, input_shape=input_shape, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics="accuracy")

model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.2)

model.evaluate(x_test, y_test)
