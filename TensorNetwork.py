import numpy as np
from Tensor import Tensor
from utils import check_input_state


class TensorNetwork:
    def __init__(self, n_qubits: int, state=None):
        self.n_qubits: int = n_qubits
        self.tensors: [Tensor] = []

        if state is not None:
            self.initialize(state)

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

    @property
    def n_qubits(self):
        return self.n_qubits

    @n_qubits.setter
    def n_qubits(self, n: int):
        assert n > 0
        self.n_qubits = n

    def initialize(self, state: str):
        assert len(state) == self.n_qubits
        assert check_input_state(state)

        for i in range(self.n_qubits):
            if state[i] == "0":
                self.tensors[i].append(Tensor.qubit(0))
            else:
                self.tensors[i].append(Tensor.qubit(1))

    def hadamard(self, qbit: int):
        """Applies hadamard gate through tensor contraction to a given qubit"""
        assert qbit in range(self.n_qubits)

        gate = Tensor.gate("H")
        self[qbit] = np.einsum("lk, kij -> lij", gate, self[qbit])

        return self

    def c_phase(self, c_qbit: int, t_qbit: int):
        ...
        return self

    def make_MPO(self):
        return self

