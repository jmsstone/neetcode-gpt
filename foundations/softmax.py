import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z,axis=0)
        f = np.vectorize(lambda x: np.e**(x-max_val))
        total = np.sum(f(z), axis=0)
        sm = np.vectorize(lambda x: round((np.e**(x-max_val))/total,4))
        return sm(z)
