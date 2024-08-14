import numpy as np
from Tensor import Tensor
from MatrixProductState import MPS


class MPO:
    physical_bond = 2
    n_bonds = 2

    ## TODO: Needed ??
    @classmethod
    def phase_MPO(cls, *args):
        return cls(*args)

    ## TODO:
    @classmethod
    def QFT_MPO(cls, *args):
        return cls(*args)

    ## TODO:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.tensors = [[] for _ in range(n_qubits)]

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
        assert self.check_input_mps(mps)
        return mps

    def __repr__(self):
        return f"MPO({self.n_qubits})\n---------------\n"

    def __len__(self):
        return self.n_qubits

    def __getitem__(self, item):
        return self.tensors[item]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def check_input_mps(self, mps: MPS) -> True:
        if len(self) == len(mps):
            return True
        else:
            return False
