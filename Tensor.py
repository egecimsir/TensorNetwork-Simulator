import numpy as np
from Utils import *


class Tensor(np.ndarray):
    """
    # Nodes in a Network
    # Inheritance of np.ndarray
    ?? Could be implemented from scratch ??
    """
    def __new__(cls, arr):
        obj = np.asarray(arr, dtype=complex)
        return obj

    @classmethod
    def qubit(cls, state: [0, 1]):
        """Creates a qubit of given state"""
        arr = np.eye(2)[state].reshape(2, 1, 1)
        return cls(arr)



