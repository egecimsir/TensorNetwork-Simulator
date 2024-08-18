import numpy as np

from Tensor import Tensor
from MatrixProductState import MPS
from utils import truncate_USV, check_input_state
from typing import Optional


class TensorNetwork:

    @classmethod
    def QFT(cls, state: str, bond_dim: Optional[int] = None):  ## fixme
        """
         QFT Tensor Network with Gate-Ansatz
        -------------------------------------
        :param state: Initial state to be fourier transformed
        :param bond_dim: Maximum bond dimension in the network
        :return: TensorNetwork with QFT applied
        """
        assert check_input_state(state)

        n_qubits = len(state)
        qc = cls(n_qubits, state)

        for i in range(n_qubits):
            qc.hadamard(i)
            for j in range(i + 1, n_qubits):
                if i + 1 != j:  ## If not adjacent
                    qc.make_adjacent(i, j, bond_dim)

                qc.c_phase(i, j, phase=np.pi / 2**j, bond_dim=bond_dim)

                if i + 1 != j:  ## If not adjacent
                    qc.restore_order(i, j, bond_dim)

        return qc

    def __init__(self, n_qubits: int, state: Optional[str] = None):
        self.n_qubits: int = n_qubits
        self.mps: MPS = MPS(n_qubits)
        if state is not None:
            self.initialize(state)

        ## Tracking
        self.ops_applied = 0
        self.ops_log = []

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
        """
        Initializes the MPS of the network with given state.
        """
        return self.mps.initialize(state)

    def retrieve_amplitude_of(self, state: str):
        """
        Retrieves the amplitude of the given state on the MPS.
        """
        return self.mps.retrieve_amplitude_of(state=state)

    def apply_single_gate(self, op: str, qbit: int, param: Optional[float] = None):
        """
        Applies 1-qubit operations on the given qubit.
        ----------------------------------------------
        :param op: Name of the operation
        :param qbit: Site of the qubit
        :param param: Rotation on the specified axis
        :return: TensorNetwork with the operation applied
        """
        assert qbit in range(self.n_qubits)

        GATE = Tensor.gate(op=op, param=param)

        if self[qbit].ndim == 2:  ## Edge Tensor
            self[qbit] = Tensor(np.einsum("lk, kj -> lj", GATE, self[qbit]))
        elif self[qbit].ndim == 3:  ## Middle Tensor
            self[qbit] = Tensor(np.einsum("lk, ikj -> ilj", GATE, self[qbit]))

        ## Tracking
        self.ops_log.append((GATE.name, qbit, param))
        self.ops_applied += 1

        return self

    def apply_multi_gate(self,
                         op: str,
                         c_qbit: int,
                         t_qbit: int,
                         param: Optional[float] = None,
                         bond_dim: Optional[int] = None):
        """
        Applies 2-qubit operations on two neighbouring qubits.
        ------------------------------------------------------
        :param op: Name of the operation
        :param c_qbit: Control qubit
        :param t_qbit: Target qubit
        :param param: Parameter value of the operation
        :param bond_dim: Maximum bond dimension of matrices not to be exceeded
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

            ## Reshape and combine U-S
            bond = S.shape[0]

            U = U.reshape(p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2, b2)

            ## Truncate U, S, V
            if bond_dim is not None:
                U, S, V = truncate_USV(bond_dim, U, S, V)

            ## Embed S into U
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

            ## Reshape and combine U-S
            bond = S.shape[0]

            U = U.reshape(b1, p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2)

            ## Truncate U, S, V
            if bond_dim is not None:
                U, S, V = truncate_USV(bond_dim, U, S, V)

            ## Embed S into U
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

            ## Reshape and combine U-S
            bond = S.shape[0]

            U = U.reshape(b1, p1, bond)
            S = np.diag(S)
            V = V.reshape(bond, p2, b2)

            ## Truncate U, S, V
            if bond_dim is not None:
                U, S, V = truncate_USV(bond_dim, U, S, V)

            ## Embed S into U
            U = np.einsum("ijk, kk -> ijk", U, S)

        ## Assign resulting tensors back to qubits
        self[c_qbit], self[t_qbit] = Tensor(U), Tensor(V)

        ## Tracking
        self.ops_log.append((GATE.name, c_qbit, t_qbit, param, bond_dim))
        self.ops_applied += 1

        return self

    def make_adjacent(self, c_qbit: int, t_qbit: int, bond_dim: Optional[int] = None):
        """
        Brings target tensor near to the control tensor by applying series of swaps.
        ----------------------------------------------------------------------------
        :param c_qbit: Tensor to be fixed at its position
        :param t_qbit: Tensor to bring backwards
        :param bond_dim: max bond_dim for swap operations
        """
        assert c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)
        assert c_qbit < t_qbit

        for i in range(t_qbit, c_qbit + 1, -1):
            print(i)
            self.swap(i - 1, i, bond_dim)

    def restore_order(self, c_qbit: int, t_qbit: int, bond_dim: Optional[int] = None):
        """
        Brings the tensor next of c_qbit back to site t_qbit by applying series of swaps.
        ---------------------------------------------------------------------------------
        :param c_qbit: Tensor to be fixed at its position
        :param t_qbit: Tensor to be moved
        :param bond_dim: max bond_dim for swap operations
        """
        assert c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)
        assert c_qbit < t_qbit

        for i in range(c_qbit + 1, t_qbit):
            self.swap(i, i + 1, bond_dim)

    def x(self, qbit: int, param: Optional[float] = None):
        """
        Applies X gate to the MPS through tensor contraction to a given qubit.
        ----------------------------------------------------------------------
        :param qbit: Site of the qubit
        :param param: Rotation on the axis
        :return: TensorNetwork
        """
        return self.apply_single_gate(op="X", qbit=qbit, param=param)

    def y(self, qbit: int, param: Optional[float] = None):
        """
        Applies Y gate to the MPS through tensor contraction to a given qubit.
        ----------------------------------------------------------------------
        :param qbit: Site of the qubit
        :param param: Rotation on the axis
        :return: TensorNetwork
        """
        return self.apply_single_gate(op="Y", qbit=qbit, param=param)

    def z(self, qbit: int, param: Optional[float] = None):
        """
        Applies Z gate to the MPS through tensor contraction to a given qubit.
        ----------------------------------------------------------------------
        :param qbit: Site of the qubit
        :param param: Rotation on the axis
        :return: TensorNetwork
        """
        return self.apply_single_gate(op="Z", qbit=qbit, param=param)

    def hadamard(self, qbit: int):
        """
        Applies hadamard gate to the MPS through tensor contraction to a given qubit.
        """
        return self.apply_single_gate(op="H", qbit=qbit, param=None)

    def swap(self, c_qbit: int, t_qbit: int, bond_dim: Optional[int] = None):
        """
        Applies swap operation on two neighbouring qubits.
        """
        return self.apply_multi_gate(op="SWAP", c_qbit=c_qbit, t_qbit=t_qbit, param=None, bond_dim=bond_dim)

    def c_not(self, c_qbit: int, t_qbit: int, bond_dim: Optional[int] = None):
        """
        Applies CNOT gate to the MPS for neighbouring qubits.
        """
        return self.apply_multi_gate(op="X", c_qbit=c_qbit, t_qbit=t_qbit, param=None, bond_dim=bond_dim)

    def c_phase(self, c_qbit: int, t_qbit: int, phase: float, bond_dim: Optional[int] = None):
        """
        Applies controlled phase gate to the MPS for neighbouring qubits.
        """
        return self.apply_multi_gate(op="Z", c_qbit=c_qbit, t_qbit=t_qbit, param=phase, bond_dim=bond_dim)
