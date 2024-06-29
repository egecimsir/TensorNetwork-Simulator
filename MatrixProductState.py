import numpy as np
from Tensor import Tensor


class MPS(Tensor):

    physical_bond = 1
    n_bonds = 2

    def __init__(self, n_qubits, bond_dims):
        super(MPS, self).__init__()






