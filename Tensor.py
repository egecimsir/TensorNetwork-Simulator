import numpy as np


class Tensor:

    physical_bond: int
    n_bonds: int

    @classmethod
    def qubit(cls, state: int | iter):
        if type(state) is int:
            assert state in [0, 1]
            data = np.eye(2, dtype=complex)[state]
        else:
            assert len(state) == 2
            assert np.linalg.norm(state) == 1
            data = np.asarray(state, dtype=complex)

        return cls(data, bond_dim=1)

    def __init__(self, data, bond_dim):
        self.array = np.asarray(data, complex)
        self._bond_dim = bond_dim

        ## Network structure
        self._position: int = 1

        self._is_leaf = True
        self._previous: Tensor
        self._next: Tensor

        self._l_bond: int
        self._r_bond: int

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"Tensor{self._position}(shape={self.array.shape} bond_dim={self.bond_dim})"

    def __str__(self):
        return str(self.array)

    def __array__(self, dtype=complex):
        """Enables interoperability with numpy"""
        return np.asarray(self.array, dtype=dtype)


    @property
    def bond_dim(self):
        return self._bond_dim

    @bond_dim.setter
    def bond_dim(self, dim):
        self._bond_dim = dim

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def reshape(self, *shape):
        self.array = self.array.reshape(shape)
        return self

