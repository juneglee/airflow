import tensorflow as tf
import argparse
import numpy as np
import logging


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
parser.add_argument(
    "--log_path",
    help="Path to log file",
    required=True,
    type=str,
)
args = parser.parse_args()
data_path = args.data_path
model_path = args.model_path
log_path = args.log_path

f = np.load(data_path)
test_x, test_y = \
    f["test_x"], f["test_y"]
test_x = test_x / 255.0

model = tf.keras.models.load_model(model_path)

loss, acc = model.evaluate(test_x, test_y)
print(f"-----model----\nloss: {loss:.4f} acc: {acc:.4f}")


logger = logging.getLogger("airflow-mnist")

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_path)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.info(f"model, {model_path}")
logger.info(f"loss, {loss}")
logger.info(f"acc, {acc}")