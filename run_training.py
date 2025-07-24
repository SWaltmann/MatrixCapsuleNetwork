from models.model import MatrixCapsuleNetwork
from utils.dataset import Dataset
import tensorflow as tf


try:
    network = MatrixCapsuleNetwork("experiments/original_model")

    network.train()
except Exception as e:
    print(e)

