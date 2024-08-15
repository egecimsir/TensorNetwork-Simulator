import numpy as np

from Tensor import Tensor
from utils import check_input_state
from MatrixProductState import MPS
from typing import Optional


class TensorNetwork:
    def __init__(self, n_qubits: int, state: Optional[str] = None):
        self.n_qubits: int = n_qubits
        self.mps: MPS = MPS(n_qubits)

        if state is not None:
            self.initialize(state)

    def __getitem__(self, item):
        return self.mps[item]

    def __setitem__(self, key, value):
        self.mps[key] = value

    def __len__(self):
        return len(self.mps)

    def __iter__(self):
        self.index = 0
        return self.index

    def __next__(self):
        if self.index < len(self.mps.tensors):
            t = self.mps.tensors[self.index]
            self.index += 1
            return t
        else:
            raise StopIteration

    def __repr__(self):
        st = f"{self.__class__.__name__}({self.n_qubits})\n---------------\n"
        for i, tensor in enumerate(self.mps.tensors):
            st += f"T{i}_{tensor.shape}\n"
        return st

    def initialize(self, state: str):
        """Apply X gates to the MPS to initialize a given state."""
        assert len(state) == self.n_qubits
        assert check_input_state(state)
        X = Tensor.gate("X")

        for s in range(self.n_qubits):
            if state[s] == "0":
                continue
            else:
                if self[s].ndim == 2:  ## Edge Tensor
                    self[s] = Tensor(np.einsum("lk, ki -> li", X, self[s]))
                elif self[s].ndim == 3:  ## Middle Tensor
                    self[s] = Tensor(np.einsum("lk, kij -> lij", X, self[s]))

        return self

    def hadamard(self, qbit: int):
        """Applies hadamard gate to the MPS through tensor contraction to a given qubit."""
        assert qbit in range(self.n_qubits)
        H = Tensor.gate("H")

        if self[qbit].ndim == 2:    ## Edge Tensor
            self[qbit] = Tensor(np.einsum("lk, ki -> li", H, self[qbit]))
        elif self[qbit].ndim == 3:  ## Middle Tensor
            self[qbit] = Tensor(np.einsum("lk, kij -> lij", H, self[qbit]))

        return self

    def cnot(self, c_qbit: int, t_qbit: int):
        """Applies controlled phase gate to the MPS through tensor contraction to the given qubits."""
        ## TODO
        return self

    def c_phase(self, c_qbit: int, t_qbit: int):
        """Applies controlled phase gate to the MPS through tensor contraction to the given qubits."""
        ## TODO
        return self

    def make_MPO(self):
        return self

