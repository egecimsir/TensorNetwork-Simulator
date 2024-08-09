import numpy as np
import einops
from Tensor import Tensor
from MatrixProductState import MPS


class MPO:

    @classmethod
    def from_circuit(cls, *args):
        ## TODO
        return cls(*args)

    def __init__(self, n_qubits, bond_dims=None):
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
        self.check_input_mps(mps)

        for i, tensor in enumerate(mps):
            ## TODO
            self.apply_transformation(tensor)

        return mps

    def __repr__(self):
        return f"MPO({self.n_qubits})\n---------------\n"

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def check_input_mps(self, mps: MPS) -> True:
        if len(self) == len(mps):
            return True
        else:
            return False

    def apply_transformation(self, t: Tensor):
        pass