import numpy as np
from Tensor import Tensor
from typing import List


class QFTMPO:
    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.sites: List[List] = [[] for _ in range(n_qubits)]


    def __len__(self):
        return self.n_qubits

    def __getitem__(self, item):
        return self.sites[item]

    def __repr__(self):
        st = ""
        for lst in self.sites:
            st += f"{str(lst)}\n"
        return st

    def initialize_qft(self):
        ## For each site (qubit) in the network:
        for i in range(self.n_qubits):
            ## Add hadamard tensor
            self.sites[i].append(Tensor.gate("H").name)

            ## Add copy tensor
            if i != self.n_qubits-1:
                self.sites[i].append(Tensor.copy_tensor().name)

            ## Add phase gates to sites below
            phase = 2
            for j in range(i+1, self.n_qubits):
                self.sites[j].append(Tensor.phase_tensor(phase).name)
                phase *= 2

        return self

    def zip_up(self):
        self.put_phase_mpo(0)
        self.contract_site(0)

        ## Add new phase mpo
        for s in range(1, self.n_qubits):
            self.put_phase_mpo(s)

            ## Contract and SVD upwards
            for i in range(self.n_qubits-1, s-1, -1):
                self.contract_site(i)


            ## SVD and contract downwards
            for i in range(s, self.n_qubits):
                ## TODO
                break

        ## Add last hadamard and contract
        self.sites[self.n_qubits-1].append(Tensor.gate("H"))
        self.contract_site(self.n_qubits-1)

        return self


    def put_phase_mpo(self, site: int):
        assert site in range(self.n_qubits-1)
        ## Add hadamard tensor
        self.sites[site].append(Tensor.gate("H").name)

        ## Add copy tenor
        self.sites[site].append(Tensor.copy_tensor().name)

        ## Add phase tensors to sites below
        phase = 2
        for s in range(site+1, self.n_qubits):
            self.sites[s].append(Tensor.phase_tensor(phase=phase).name)
            phase *= 2

        return self

    def contract_site(self, site: int):
        assert site in range(self.n_qubits)
        ## TODO
        pass
