import numpy as np
from Tensor import Tensor


class TensorNetwork:
    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.tensors: [[Tensor]] = []

    def __getitem__(self, item):
        return self.tensors[item]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

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

    def __repr__(self):
        return str(self.tensors)

    def hadamard(self, qbit: int):
        return self

    def c_phase(self, c_qbit: int, t_qbit: int):
        return self

    def make_MPO(self):
        return self

