import numpy as np
from Tensor import Tensor
from MatrixProductState import MPS


class MPO:
    physical_bond = 2
    n_bonds = 2

    ## TODO:
    @classmethod
    def from_circuit(cls, *args):
        return cls(*args)

    ## TODO:
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

    ## TODO:
    def __call__(self, mps: MPS) -> MPS:
        """
        Multiplies the given MPS with the MPO.
        --------------------------------------
        :param mps: MPS to be evolved.
        :return: evolved MPS
        """
        self.check_input_mps(mps)
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
