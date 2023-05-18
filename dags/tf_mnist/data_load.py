import tensorflow as tf
import os
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_path",
    help="Path to mnist data",
    required=True,
    type=str,
)
parser.add_argument(
    "--test_data_path",
    help="Path to mnist data",
    required=True,
    type=str,
)
args = parser.parse_args()
train_data_path = args.train_data_path
test_data_path = args.test_data_path

os.makedirs(Path(train_data_path).parent.absolute(), exist_ok=True)
os.makedirs(Path(test_data_path).parent.absolute(), exist_ok=True)

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

np.savez(
    train_data_path,
    train_x=train_x,
    train_y=train_y
)
print(f"save train_data in :{train_data_path}")
np.savez(
    test_data_path,
    test_x=test_x,
    test_y=test_y
)
print(f"save test_data in :{test_data_path}")