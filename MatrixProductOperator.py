import numpy as np
from Tensor import Tensor
from MatrixProductState import MPS


class MPO:

    def __init__(self, n_qubits, bond_dims):
        self.n_qubits = n_qubits
        self.bond_dims = bond_dims
        self.tensors = []

        for i in range(n_qubits):
            if i == 0:
                pass
            elif i == n_qubits:
                pass
            else:
                pass

    def __call__(self, mps: MPS) -> MPS:
        assert len(self) == len(mps)

        for i, tensor in enumerate(mps):
            tensor = np.einsum("", self[i], )


        return mps

    def __repr__(self):
        return f"MPO({self.n_qubits})"

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]

    def __setitem__(self, key, value):
        self.tensors[key] = value

