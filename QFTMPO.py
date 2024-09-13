import numpy as np

from MatrixProductState import MPS
from Tensor import Tensor
from typing import List


class QFTMPO:
    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.sites: List[List[Tensor]] = [[] for _ in range(n_qubits)]


    def __call__(self, mps: MPS):
        """
        MPO-MPS multiplication.
        -----------------------
        """
        assert len(mps) == len(self)
        pass

    def __len__(self):
        return self.n_qubits

    def __getitem__(self, item):
        return self.sites[item]

    def __setitem__(self, key, value):
        self.sites[key] = value

    def __repr__(self):
        st = ""
        for i, lst in enumerate(self.sites):
            names = [t.name for t in lst]
            st += f"s{i}: {str(names)}\n"
        return st

    def initialize_qft(self):
        """
        Initializes sites with tensors according to QFT schema.
        -------------------------------------------------------
        """
        ## For each site (qubit) in the network:
        for i in range(self.n_qubits):
            ## Add hadamard tensor
            self.sites[i].append(Tensor.gate("H"))

            ## Add copy tensor
            if i != self.n_qubits-1:
                self.sites[i].append(Tensor.copy_tensor())

            ## Add phase gates to sites below
            phase = np.pi / 2
            for j in range(i+1, self.n_qubits):
                P = Tensor.phase_tensor(phase=phase, ndim=4 if j != len(self) - 1 else 3)
                self.sites[j].append(P)
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
                if i != s:  ## Don't apply SVD for first site of PhaseMPO.

                    T = self.sites[i].pop(0)

                    ## Reshape Tensor into order-2 for the SVD
                    if T.ndim == 4:
                        v1, v2, h1, h2 = T.shape
                        T.reshape(v1*v2, h1*h2)

                    elif T.ndim == 5:
                        v1, v2, v3, h1, h2 = T.shape
                        T.reshape(v1*v2*v3, h1*h2)

                    else:
                        raise ValueError

                    U, S, V = np.linalg.svd(T, full_matrices=False)
                    U = np.einsum("ij, jj -> ij", U, np.diag(S))

                    self.sites[i] = [Tensor(V)]
                    self.sites[i-1].append(Tensor(U, "U"))

            ## SVD and contract downwards
            for i in range(s, self.n_qubits):
                ## TODO
                break

        ## Add last hadamard and contract
        self.sites[self.n_qubits-1].append(Tensor.gate("H"))
        self.contract_site(self.n_qubits-1)

        return self

    def put_phase_mpo(self, site: int):
        """
        Appends Hadamard and Copy tensors to the given site, Phase tensors all the sites below.
        ---------------------------------------------------------------------------------------
        """
        assert site in range(self.n_qubits-1)
        ## Add hadamard tensor
        self.sites[site].append(Tensor.gate("H"))

        ## Add copy tenor
        self.sites[site].append(Tensor.copy_tensor())

        ## Add phase tensors to sites below
        phase = np.pi / 2
        for s in range(site+1, self.n_qubits):
            P = Tensor.phase_tensor(phase=phase, ndim=4 if s != len(self)-1 else 3)
            self.sites[s].append(P)
            phase /= 2

        return self

    def contract_site(self, site: int):
        """
        Contracts all the tensors in a given site until single tensor remains.
        ----------------------------------------------------------------------
        :param site: site to be contracted
        """
        assert site in range(self.n_qubits)

        ## First site
        if site == 0:
            H, C = self.sites[site]
            T = np.einsum("ij, kjb -> kib", H, C)

        ## Last site
        elif site == self.n_qubits-1:
            T1, T2 = self.sites[site]
            if T1.ndim == T2.ndim:
                T = np.einsum("kij, ajb -> kaib", T1, T2)  ## T3--P3
            else:
                T = np.einsum("kij, jb -> kib", T1, T2)  ## T3--H2

        ## Middle sites
        else:
            if len(self.sites[site]) == 3:  # PhaseMPO below:
                T4, P, U = self.sites[site]
                T = np.einsum("abij, cdjk, bdm -> acmik", T4, P, U)  ## T4--P4--U3

            elif len(self.sites[site]) == 4:  # PhaseMPO begin:
                T4, H, C, U = self.sites[site]
                T = np.einsum("abij, jk, ckl, bcm -> amil", T4, H, C, U)  ## T4--H2--C3--U3

            else:
                raise ValueError

        ## Replace site with new tensor
        self.sites[site] = [Tensor(T)]

