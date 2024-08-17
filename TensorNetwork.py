import numpy as np

from Tensor import Tensor
from utils import check_input_state
from MatrixProductState import MPS
from typing import Optional


class TensorNetwork:
    def __init__(self, n_qubits: int, state: Optional[str] = None):
        self.n_qubits: int = n_qubits
        self.mps: MPS = MPS(n_qubits)

        if state is not None:
            self.initialize(state)

    def __getitem__(self, item):
        return self.mps[item]

    def __setitem__(self, key, value):
        self.mps[key] = value

    def __len__(self):
        return len(self.mps)

    def __iter__(self):
        self.index = 0
        return self.index

    def __next__(self):
        if self.index < len(self.mps.tensors):
            t = self.mps.tensors[self.index]
            self.index += 1
            return t
        else:
            raise StopIteration

    def __repr__(self):
        st = f"{self.__class__.__name__}({self.n_qubits})\n---------------\n"
        for i, tensor in enumerate(self.mps.tensors):
            st += f"T{i}_{tensor.shape}\n"
        return st

    def initialize(self, state: str):
        return self.mps.initialize(state)

    def retrieve_amplitude_of(self, state: str):
        """
        Retrieves the amplitude of the given state
        """
        return self.mps.retrieve_amplitude_of(state=state)

    def apply_single_gate(self, op: str, qbit: int, param: Optional[float] = None):
        assert qbit in range(self.n_qubits)

        GATE = Tensor.gate(op=op, param=param)

        if self[qbit].ndim == 2:  ## Edge Tensor
            self[qbit] = Tensor(np.einsum("lk, kj -> lj", GATE, self[qbit]))
        elif self[qbit].ndim == 3:  ## Middle Tensor
            self[qbit] = Tensor(np.einsum("lk, ikj -> ilj", GATE, self[qbit]))

        return self

    def apply_multi_gate(self, op: str, c_qbit: int, t_qbit: int, param: Optional[float] = None):
        """
        Applies 2-qubit operations on two neighbouring qubits
        -----------------------------------------------------
        :param op: Name of the operation
        :param c_qbit: Control qubit
        :param t_qbit: Target qubit
        :param param: Parameter value of the operation
        :return: TensorNetwork with the operation applied
        """
        assert c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)
        assert c_qbit + 1 == t_qbit

        GATE = Tensor.c_gate(op=op, param=param)
        Mc, Mt = self[c_qbit], self[t_qbit]

        if c_qbit == 0:
            ## Contract both tensors to a 3d-Tensor
            T1 = np.einsum("ij, jlk -> ilk", Mc, Mt)

            ## Contract Tensor with 4d-Tensor Swap Gate
            T2 = np.einsum("vwil, ilk -> vwk", GATE, T1)

            ## Singular Value Decomposition
            p1, p2, b2 = T2.shape
            T2 = T2.reshape(p1, p2 * b2)
            U, S, V = np.linalg.svd(T2, full_matrices=False)

            ## Adjust bond_dim (w/o truncation)
            # TODO: implement truncation
            bond = S.shape[0]

            ## Reshape and combine U-S
            U = U.reshape(p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2, b2)
            U = np.einsum("ij, jj -> ij", U, S)

        elif t_qbit == self.n_qubits - 1:
            ## Contract both tensors to a 3d-Tensor
            T1 = np.einsum("jik, kl -> jil", Mc, Mt)

            ## Contract Tensor with 4d-Tensor Swap Gate
            T2 = np.einsum("vwil, jil -> jvw", GATE, T1)

            ## Singular Value Decomposition
            b1, p1, p2 = T2.shape
            T2 = T2.reshape(b1 * p1, p2)
            U, S, V = np.linalg.svd(T2, full_matrices=False)

            ## Adjust bond_dim (w/o truncation)
            # TODO: implement truncation
            bond = S.shape[0]

            ## Reshape and combine U-S
            U = U.reshape(b1, p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2)
            U = np.einsum("jik, kk -> jik", U, S)

        else:
            ## Contract both tensors to a 4d-Tensor
            T1 = np.einsum("jik, klm -> jilm", Mc, Mt)

            ## Contract Tensor with 4d-Tensor Swap Gate
            T2 = np.einsum("vwil, jilm -> jvwm", GATE, T1)

            ## Singular Value Decomposition
            b1, p1, p2, b2 = T2.shape
            T2 = T2.reshape(p1 * b1, p2 * b2)
            U, S, V = np.linalg.svd(T2, full_matrices=False)

            ## Adjust bond_dim (w/o truncation)
            # TODO: implement truncation
            bond = S.shape[0]

            ## Reshape and combine U-S
            U = U.reshape(b1, p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2, b2)
            U = np.einsum("ijk, kk -> ijk", U, S)

        ## Assign resulting tensors back to qubits
        self[c_qbit], self[t_qbit] = Tensor(U), Tensor(V)

        return self

    def hadamard(self, qbit: int):
        """
        Applies hadamard gate to the MPS through tensor contraction to a given qubit.
        """
        return self.apply_single_gate(op="H", qbit=qbit, param=None)

    def x(self, qbit: int, param: Optional[float] = None):
        return self.apply_single_gate(op="X", qbit=qbit, param=param)

    def y(self, qbit: int, param: Optional[float] = None):
        return self.apply_single_gate(op="Y", qbit=qbit, param=param)

    def z(self, qbit: int, param: Optional[float] = None):
        return self.apply_single_gate(op="Z", qbit=qbit, param=param)

    def swap(self, c_qbit: int, t_qbit: int):
        """
        Applies swap operation on two neighbouring qubits
        """
        return self.apply_multi_gate(op="SWAP", c_qbit=c_qbit, t_qbit=t_qbit, param=None)

    def cnot(self, c_qbit: int, t_qbit: int):
        """
        Applies controlled phase gate to the MPS through tensor contraction to two neighbouring qubits.
        """
        return self.apply_multi_gate(op="X", c_qbit=c_qbit, t_qbit=t_qbit, param=None)

    def c_phase(self, c_qbit: int, t_qbit: int, phase: float):
        """
        Applies controlled phase gate to the MPS through tensor contraction to two neighbouring qubits.
        """
        return self.apply_multi_gate(op="Z", c_qbit=c_qbit, t_qbit=t_qbit, param=phase)
