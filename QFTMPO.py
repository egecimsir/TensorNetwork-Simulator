import numpy as np
from Tensor import Tensor
from typing import List


class QFTMPO:
    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.gates: List[List] = [[] for _ in range(n_qubits)]  # TODO: List[List[Tensor]]

        self.initialize_qft()

    def __len__(self):
        return self.n_qubits

    def __getitem__(self, item):
        return self.gates[item]

    def __repr__(self):
        st = ""
        for lst in self.gates:
            st += f"{str(lst)}\n"
        return st

    def initialize_qft(self):
        ## For each site (qubit) in the network:
        for i in range(self.n_qubits):
            ## Add hadamard tensor
            self.gates[i].append("H")

            ## Add copy tensor
            if i != self.n_qubits-1:
                self.gates[i].append("Copy")

            ## Add phase gates to sites below
            phase = 2
            for j in range(i+1, self.n_qubits):
                self.gates[j].append(f"P_{phase}")
                phase *= 2

    def network_to_mpo(self):
        pass
