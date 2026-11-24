import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(np.dot(X, weights),5)

    def err(self, model_pred, ground_truth):
        return model_pred - ground_truth

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        e = np.vectorize(self.err)
        return np.round(np.mean(e(model_prediction,ground_truth)**2),5)
        
