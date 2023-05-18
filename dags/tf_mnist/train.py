import tensorflow as tf
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    help="Path to mnist data",
    required=True,
    type=str,
)
parser.add_argument(
    "--model_path",
    help="Path to mnist model",
    required=True,
    type=str,
)
args = parser.parse_args()
data_path = args.data_path
model_path = args.model_path

f = np.load(data_path)
train_x, train_y = \
    f["train_x"], f["train_y"]

train_x = train_x / 255.0

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=1)

model.save(model_path)
print(f"model saved on : {model_path}")