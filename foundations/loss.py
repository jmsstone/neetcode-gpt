import numpy as np
from numpy.typing import NDArray


class Solution:
    def bce(self, y_true, y_pred):
        return y_true * np.log(y_pred) + (1-y_true) * np.log(1- y_pred)

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = y_pred.clip(0.0000001, 0.9999999)
        f = np.vectorize(self.bce)
        return np.round(-1 * np.mean(f(y_true,y_pred), axis = 0),4)
        
    def cce(self, y_true, y_pred):
        return y_true * np.log(y_pred)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = y_pred.clip(0.0000001, 0.9999999)
        f = np.vectorize(self.cce)
        a = np.sum(f(y_true,y_pred),axis=1)
        print(a)
        return np.round(-1 * np.mean(a), 4)

