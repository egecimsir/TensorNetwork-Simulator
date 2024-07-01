import numpy as np
import utils


class Tensor:

    @classmethod
    def qubit(cls, state: int):
        assert int(state) in (0, 1)
        return cls(np.eye(2, dtype=complex)[state])

    def __init__(self, data=None):
        self.array = np.asarray(data, complex)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"Tensor{self.array.shape}"

    def __str__(self):
        return str(self.array)

    def __array__(self, dtype=complex):
        """Enables interoperability with numpy"""
        return np.asarray(self.array, dtype=dtype)

    @property
    def shape(self):
        return self.array.shape

    @property
    def bond_dim(self):
        return self.shape[-1]

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def reshape(self, *shape):
        self.array = self.array.reshape(shape)
        return self

    def conjugate(self):
        self.array = self.array.conjugate()
        return self
