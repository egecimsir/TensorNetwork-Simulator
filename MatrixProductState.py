import numpy as np

from Tensor import Tensor
from utils import check_input_state
from typing import Optional, List


class MPS:
    physical_bond = 1
    n_bonds = 2

    def __init__(self, n_qubits: int, bond_dims: Optional[List[int]] = None):
        self.n_qubits: int = n_qubits
        self.tensors: [Tensor] = []

        if bond_dims is not None:
            assert len(bond_dims) + 1 == self.n_qubits
            self.bond_dims = bond_dims
        else:
            self.bond_dims = [1 for _ in range(self.n_qubits - 1)]

        for i in range(n_qubits):
            if i == 0:
                tensor = Tensor.qubit(0).reshape(2, self.bond_dims[i])
            elif i == n_qubits - 1:
                tensor = Tensor.qubit(0).reshape(2, self.bond_dims[i - 1])
            else:
                tensor = Tensor.qubit(0).reshape(2, self.bond_dims[i - 1], self.bond_dims[i])

            self.tensors.append(tensor)

    def __repr__(self):
        st = f"{self.__class__.__name__}({self.n_qubits})\n---------------\n"
        for i, tensor in enumerate(self.tensors):
            st += f"T{i}_{tensor.shape}\n"
        return st

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def __iter__(self):
        self.index = 0
        return self.index

    def __next__(self):
        if self.index < len(self.tensors):
            t = self.tensors[self.index]
            self.index += 1
            return t
        else:
            raise StopIteration

    def retrieve_amplitude_of(self, state: str):
        assert len(state) == self.n_qubits
        assert check_input_state(state)
        tensors = []

        ## Fix physical indices
        for s in range(len(state)):
            idx = int(state[s])
            arr = self.tensors[s][idx]
            tensors.append(arr)

        ## Matrix-Vector products
        row_vec = tensors.pop(0)
        col_vec = tensors.pop(-1)
        for mat in tensors:
            row_vec = np.einsum("i, ij -> j", row_vec, mat)

        return row_vec @ col_vec
