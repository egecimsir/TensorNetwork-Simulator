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

    def __setitem__(self, key, value):
        self.sites[key] = value

    def __repr__(self):
        st = ""
        for i, lst in enumerate(self.sites):
            st += f"s{i}: {str(lst)}\n"
        return st

    def initialize_qft(self):
        """
        Initializes sites with tensors according to QFT schema.
        -------------------------------------------------------
        """
        ## For each site (qubit) in the network:
        for i in range(self.n_qubits):
            ## Add hadamard tensor
            self.sites[i].append(Tensor.gate("H").name)

            ## Add copy tensor
            if i != self.n_qubits-1:
                self.sites[i].append(Tensor.copy_tensor().name)

            ## Add phase gates to sites below
            phase = np.pi / 2
            for j in range(i+1, self.n_qubits):
                self.sites[j].append(Tensor.phase_tensor(phase).name)
                phase /= 2

        return self

    def zip_up(self):
        """
        Builds the QFT MPO using the zip-up algorithm.
        ----------------------------------------------
        """
        ## Begin with first PhaseMPO and contract the first site.
        self.put_phase_mpo(0)
        self.contract_site(0)

        ## Add new PhaseMPO one by one
        for s in range(1, self.n_qubits):
            self.put_phase_mpo(s)

            ## Contract and SVD upwards
            for i in range(self.n_qubits-1, s-1, -1):
                self.contract_site(i)

                ## Apply SVD to the site; leave V at site, push U to the site above.
                ## Don't apply SVD for first site of PhaseMPO.
                if i != s:

                    T = self.sites[i].pop(0)
                    ## TODO: reshape T?
                    U, S, V = np.linalg.svd(T, full_matrices=False)
                    U = np.einsum("ij, jj -> ij", U, np.diag(S))

                    self.sites[i].append(V)
                    self.sites[i-1].append(U)


            ## SVD and contract downwards
            for i in range(s, self.n_qubits):
                ## TODO
                break

        ## Add last hadamard and contract
        self.sites[self.n_qubits-1].append(Tensor.gate("H"))
        self.contract_site(self.n_qubits-1)

        return self

    ## TODO: Append tensors instead of their names
    def put_phase_mpo(self, site: int):
        assert site in range(self.n_qubits-1)
        ## Add hadamard tensor
        self.sites[site].append(Tensor.gate("H").name)

        ## Add copy tenor
        self.sites[site].append(Tensor.copy_tensor().name)

        ## Add phase tensors to sites below
        phase = np.pi / 2
        for s in range(site+1, self.n_qubits):
            P = Tensor.phase_tensor(phase=phase, ndim=4 if s != len(self)-1 else 3)
            self.sites[s].append(P.name)
            phase /= 2

        return self

    ## TODO: Arrange position of dimensions of needed
    def contract_site(self, site: int):
        """
        Contracts all the tensors in a given site until single tensor remains.
        ----------------------------------------------------------------------
        :param site: site to be contracted
        """
        assert site in range(self.n_qubits)

        ## Consists of: H2--C3
        if site == 0:
            H, C = self.sites[site]
            T = np.einsum("ij, jkl -> ikl", H, C)

            self.sites[site] = [T]

        ### Consists of ###
        # first: P3--P3
        # then: T3--P3
        # lastly: T3--H2
        elif site == self.n_qubits-1:
            pass

        ### Consists of ###
        # PhaseMPO begin: P4--H2--C3--U3 | T4--H2--C3--U3
        # Middle sites:  P4--P4--U3 | T4--P4--U3
        else:
            pass
