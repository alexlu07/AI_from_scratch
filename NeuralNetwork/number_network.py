from network import Network
import numpy as np
import pandas as pd


class NumberNetwork(Network):
    def __init__(self):
        super().__init__(784, 128, 128, 10)

    def train_with_dataset(self):
        ip = np.load('mnist/train-images.npy')
        ip = ip.reshape([ip.shape[0], -1])
        ip = ip.astype("float64")
        ip /= 255
        actual = pd.get_dummies(np.load('mnist/train-labels.npy'))
        actual = actual.to_numpy()
        self.train(ip, actual)

    def test_with_dataset(self):
        ip = np.load('mnist/train-images.npy')
        ip = ip.reshape([ip.shape[0], -1])
        ip = ip.astype("float64")
        ip = ip / 255
        actual = pd.get_dummies(np.load('mnist/train-labels.npy'))
        actual = actual.to_numpy()
        return self.test(ip, actual)