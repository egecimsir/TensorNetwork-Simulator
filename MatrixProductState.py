import numpy as np

from Tensor import Tensor
from utils import check_input_state
from typing import Optional, List


class MPS:
    physical_bond = 1
    n_bonds = 2

    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.tensors: [Tensor] = []

        for i in range(n_qubits):
            if i == 0:
                tensor = Tensor.qubit(0).squeeze(0)
            elif i == n_qubits - 1:
                tensor = Tensor.qubit(0).squeeze(2)
            else:
                tensor = Tensor.qubit(0)

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

    def initialize(self, state: str):
        """
        Apply X gates to the MPS to initialize a given state.
        """
        assert len(state) == self.n_qubits
        assert check_input_state(state)

        X = Tensor.gate("X")

        for s in range(self.n_qubits):
            if state[s] == "0":
                continue
            else:
                if self[s].ndim == 2:  ## Edge Tensor
                    self[s] = Tensor(np.einsum("lk, kj -> lj", X, self[s]))
                elif self[s].ndim == 3:  ## Middle Tensor
                    self[s] = Tensor(np.einsum("lk, ikj -> ilj", X, self[s]))
        return self

    def retrieve_amplitude_of(self, state: str):
        """
        Retrieves the amplitude of the given state on the MPS.
        ------------------------------------------------------
        :param state: Classical binary-string state
        :return: A complex number representing the amplitude of the state
        """
        assert len(state) == self.n_qubits
        assert check_input_state(state)

        tensors = []

        ## Fix physical indices
        for s in range(len(state)):
            idx = int(state[s])
            if s == 0:
                arr = self.tensors[s][idx, :]
            elif s == self.n_qubits - 1:
                arr = self.tensors[s][:, idx]
            else:
                arr = self.tensors[s][:, idx, :]
            tensors.append(arr)

        ## Matrix-Vector products
        row_vec = tensors.pop(0)
        col_vec = tensors.pop(-1)
        for mat in tensors:
            row_vec = np.einsum("i, ij -> j", row_vec, mat)

        return row_vec @ col_vec
