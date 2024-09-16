import numpy as np

from MatrixProductState import MPS
from Tensor import Tensor
from utils import truncate_USV
from typing import List, Optional


class QFTMPO:
    def __init__(self, n_qubits: int):
        self.n_qubits: int = n_qubits
        self.sites: List[List[Tensor]] = [[] for _ in range(n_qubits)]

    def __call__(self, mps: MPS, bond_dim: Optional[int] = None):
        """
        Multiplies the given MPS with the MPO using zip-up algorithm.
        -------------------------------------------------------------
        """
        assert len(mps) == len(self)

        ## Put mps and mpo tensors in lists
        tensors = []
        for i in range(len(mps)):
            tensors.append([mps[i], self[i][0]])

        ## Zip-up
        for i in range(len(mps)):

            if i == 0:  ## First site

                ## Contract MPS-MPO tensors at site-0
                S, M = tensors[i]
                T = np.einsum("ix, kij -> xkj", S, M)

                ## Reshape T: (j)(xk)
                x, k, j = T.shape
                T = T.reshape(j, x*k)

                ## SVD
                U, S, V = np.linalg.svd(T, full_matrices=False)
                S = np.diag(S)

                ## Truncate to bond_dim
                if bond_dim is not None:
                    bond_dim = min(len(S), bond_dim)
                    U, S, V = truncate_USV(bond_dim, U, S, V)

                ## Embed S into V
                V = np.einsum("jj, ji -> ji", S, V)

                ## Adjust dimensions
                bond = S.shape[0]
                U = U.reshape(j, bond)
                V = V.reshape(bond, x, k)

                ## Save U at site-i and append V below
                tensors[i] = U
                tensors[i+1].append(V)

            elif i == len(mps)-1:  ## Last site

                S, M, V = tensors[i]
                T = np.einsum("xi, kij, axk -> aj", S, M, V)
                tensors[i] = T

            else:  ## Middle sites

                ## Contract MPS-MPO tensors at site-i
                S, M, V = tensors[i]
                T = np.einsum("xiy, klij, axk -> ylaj", S, M, V)

                ## Reshape T: (aj)(xk)
                y, l, a, j = T.shape
                T = T.reshape(j*a, y*l)

                ## SVD
                U, S, V = np.linalg.svd(T, full_matrices=False)
                S = np.diag(S)

                ## Truncate to bond_dim
                if bond_dim is not None:
                    bond_dim = min(len(S), bond_dim)
                    U, S, V = truncate_USV(bond_dim, U, S, V)

                ## Embed S into V
                V = np.einsum("jj, ji -> ji", S, V)

                ## Adjust dimensions
                bond = S.shape[0]
                U = U.reshape(a, j, bond)
                V = V.reshape(bond, y, l)

                ## Save U at site-i and append V below
                tensors[i] = U
                tensors[i + 1].append(V)

        ## Replace MPS tensors
        mps.tensors = tensors

        return mps

    def __len__(self):
        return self.n_qubits

    def __getitem__(self, item):
        return self.sites[item]

    def __setitem__(self, key, value):
        self.sites[key] = value

    def __repr__(self):
        st = f"{self.__class__.__name__}({self.n_qubits})\n--------------------\n"
        for i, lst in enumerate(self.sites):
            names = [t.name for t in lst]
            st += f"s{i}: {str(names)}\n"
        return st

    def display_qft(self):
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

        print(self)
        self.sites = [[] for _ in range(self.n_qubits)]

    def zip_up(self, bond_dim: Optional[int] = None):
        """
        Builds the QFT MPO using the zip-up algorithm.
        ----------------------------------------------
        """
        ## Begin with first PhaseMPO and contract the first site.
        self.put_phase_mpo(0)
        self.contract_site(0)

        ## Add new PhaseMPO one by one
        for s in range(1, self.n_qubits-1):
            self.put_phase_mpo(s)

            ## Contract and SVD upwards
            for i in range(self.n_qubits-1, s-1, -1):
                self.contract_site(i)

                ## Apply SVD to the site; leave V at site, push U to the site above.
                if i != s:  ## Don't apply SVD for first site of PhaseMPO.

                    T = self.sites[i].pop(0)

                    if T.ndim == 4:
                        ## Reshape
                        v1, v2, h1, h2 = T.shape
                        T = T.reshape(v1*h1, v2*h2)

                        ## Apply SVD and embed S into U
                        U, S, V = np.linalg.svd(T, full_matrices=False)
                        S = np.diag(S)

                        ## Truncate to bond_dim
                        if bond_dim is not None:
                            bond_dim = min(len(S), bond_dim)
                            U, S, V = truncate_USV(bond_dim, U, S, V)

                        ## Embed S into U
                        U = np.einsum("ij, jj -> ij", U, S)

                        ## Adjust dimensions
                        bond = S.shape[0]
                        U = U.reshape(v1, v2, bond)
                        V = V.reshape(bond, h1, h2)

                    elif T.ndim == 5:
                        ## Reshape
                        v1, v2, v3, h1, h2 = T.shape
                        T = T.reshape(v1*v2, v3*h1*h2)

                        ## Apply SVD and embed S into U
                        U, S, V = np.linalg.svd(T, full_matrices=False)
                        S = np.diag(S)

                        ## Truncate to bond_dim
                        if bond_dim is not None:
                            bond_dim = min(len(S), bond_dim)
                            U, S, V = truncate_USV(bond_dim, U, S, V)

                        ## Embed S into U
                        U = np.einsum("ij, jj -> ij", U, S)

                        ## Adjust dimensions
                        bond = S.shape[0]
                        U = U.reshape(v1, v2, bond)
                        V = V.reshape(bond, v3, h1, h2)

                    else:
                        raise ValueError

                    ## Replace site with V and append U to above site
                    self.sites[i] = [Tensor(V, name=f"T{V.ndim}")]
                    self.sites[i-1].append(Tensor(U, name=f"U{U.ndim}"))

            ## SVD and contract downwards
            for i in range(s, self.n_qubits):
                break  ## TODO: debug

                if i == s:  ## first site of PhaseMPO
                    T = self[i][0]

                    ## Reshape
                    v1, v2, h1, h2 = T.shape
                    T = T.reshape(h1*h2*v1, v2)

                    ## SVD
                    U, S, V = np.linalg.svd(T)
                    S = np.diag(S)

                    ## Truncate to bond_dim
                    if bond_dim is not None:
                        bond_dim = min(len(S), bond_dim)
                        U, S, V = truncate_USV(bond_dim, U, S, V)

                    ## Embed S into V
                    V = np.einsum("jj, ji -> ji", S, V)

                    ## Adjust dimensions
                    bond = len(S)
                    U = U.reshape(v1, bond, h1, h2)  ## fixme
                    V = V.reshape(bond, v2)

                    ## Replace site with U and append V below
                    self[i] = [Tensor(U)]
                    self[i+1].append(V)

                elif i == len(self)-1:  ## last site
                    T, V = self[s]
                    T = np.einsum("kij, bk -> bij", T, V)
                    self[i] = [Tensor(T)]

                else:   ## middle sites

                    T, V = self[s]
                    T = np.einsum("klij, bk -> blij", T, V)

                    ## Reshape
                    v1, v2, h1, h2 = T.shape
                    T.reshape(h1*h2*v1, v2)

                    ## SVD
                    U, S, V = np.linalg.svd(T)
                    S = np.diag(S)

                    ## Truncate to bond_dim
                    if bond_dim is not None:
                        bond_dim = min(len(S), bond_dim)
                        U, S, V = truncate_USV(bond_dim, U, S, V)

                    ## Embed S into V
                    V = np.einsum("jj, ji -> ji", S, V)

                    ## Adjust dimensions
                    bond = len(S)
                    U = U.reshape(v1, bond, h1, h2)
                    V = V.reshape(bond, v2)

                    ## Replace site with U and append V below
                    self[i] = [Tensor(U)]
                    self[i + 1].append(V)


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
        self.sites[site] = [Tensor(T, name=f"T{T.ndim}")]

    def print_dims(self):
        print(f"{self.__class__.__name__}({self.n_qubits})\n--------------------\n")
        for i, lst in enumerate(self.sites):
            print(f"s{i}: {str([t.shape for t in lst])}")
