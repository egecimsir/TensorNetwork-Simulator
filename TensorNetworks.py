import numpy as np
from Tensor import Tensor


class MPS:
    """
    Tensor Representation of Quantum Circuit
    """
    SingleGates = {
        "X": Tensor([[0, 1], [1, 0]]),
        "Y": Tensor([[0, -1j], [1j, 0]]),
        "Z": Tensor([[1, 0], [0, -1]]),
        "H": Tensor([[1, 1], [1, -1]]) / 2 ** (1 / 2)
    }

    MultiGates = {
        "CX": Tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]]).reshape(2, 2, 2, 2),
        "SWAP": Tensor([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]]).reshape(2, 2, 2, 2)
    }

    ########
    ### TODO: Research & Implement the TEBD Algorithm
    ### TODO: Implement the getAmplitude method
    ########


    def __init__(self, num_qubits: int, bond_dim: int, state=None):
        self.tensors: np.ndarray
        self.n_qubits = num_qubits
        self.bond_dim = bond_dim

        if state is None:
            self.tensors = np.array([Tensor.qubit(0) for t in range(num_qubits)])
        else:
            self.initialize(state)

        ## Tracking
        ## TODO: Track bond_dims of qubits
        self.gates_applied = []

    def __getitem__(self, item: int):
        assert item in range(len(self.tensors))
        return self.tensors[item]

    def __setitem__(self, item: int, value):
        assert item in range(len(self.tensors))
        self.tensors[item] = value

    def __add__(self, other):
        ## TODO: ?? Extend circuit vertically with another circuit ??
        pass

    def __mul__(self, other):
        ## TODO: ?? Multiply circuit qbit by qbit with another circuit of the same size ??
        pass

    def __repr__(self):
        ## TODO: Include __dict__
        st = f"QuantumCircuit({self.n_qubits})\n"
        for q in self.tensors:
            __ = f"\n{str(q)}\n\n"
            st = st + __
        return st

    def __str__(self):
        ## TODO: Print only string of the tensors vertically
        return "\n".join([str(q) for q in self.tensors])

    @classmethod
    def createControlledUnitary(cls, gate: str, add2dict=True):
        assert gate in cls.SingleGates.keys()

        mat = np.eye(4, dtype=complex)
        mat[:2, :2] = MPS.SingleGates[gate]
        if add2dict:
            name = "C" + gate
            cls.MultiGates[name] = mat
        return mat.reshape(2, 2, 2, 2)

    def initialize(self, arr: [bool]):
        """
        Initialize the circuit with a given bit string/list
        """
        assert len(arr) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, arr))), "Initial states must be 0 or 1"

        arr = list(map(int, arr))
        self.tensors = [Tensor.qubit(arr[i]) for i in range(self.n_qubits)]

    def set_state(self, arr: [bool]):
        """
        Selecting the corresponding state dimensions (i_n = {0,1})
        of the tensor network
        """
        assert len(arr) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, arr))), "States must be 0 or 1"

        arr = list(map(int, arr))
        for i, state in enumerate(arr):
            self.tensors[i] = self.tensors[i][state]

    def applySingleGate(self, op: SingleGates, qbit: int):
        assert qbit in range(len(self.tensors)), "Qbit not found in tensors"
        assert op in MPS.SingleGates.keys(), f"Unrecognized operation: {str(op)}"

        gate = MPS.SingleGates[op]
        tensor = self.tensors[qbit]
        self.tensors[qbit] = np.einsum("lk, kij -> lij", gate, tensor)

        ## Keep record & End of method
        self.gates_applied.append((qbit, op))

    def applyRotational(self, op, theta: float, qbit: int):
        assert qbit in range(len(self.tensors)), "Qbit not found in tensors"
        assert op in ["X", "Y", "Z"], "Unrecognized operation"

        # theta = theta % (4 * np.pi)
        sin = np.sin
        cos = np.cos
        exp = np.exp
        gate = 0

        if op == "X":
            gate = np.array([[cos(theta/2), -1j*sin(theta/2)],
                             [-1j*sin(theta/2), cos(theta/2)]], dtype=complex)
        if op == "Y":
            gate = np.array([[cos(theta/2), -sin(theta/2)],
                             [sin(theta/2), cos(theta/2)]], dtype=complex)
        if op == "Z":
            gate = np.array([[exp(-1j*theta/2), 0],
                             [0, exp(1j*theta/2)]], dtype=complex)

        tensor = self.tensors[qbit]
        self.tensors[qbit] = np.einsum("lk, kij -> lij", gate, tensor)

        ## Keep record & End of method
        self.gates_applied.append((qbit, op, theta))

    def applyMultiGate(self, op: MultiGates, c_qbit: int, t_qbit: int):
        assert 0 <= c_qbit and t_qbit <= self.n_qubits, "Qbit index does not match"
        assert t_qbit == c_qbit + 1, "Qbits must be neighbors"
        assert op in MPS.MultiGates.keys(), f"Unrecognized operation: {str(op)}"

        ## Creating 4d-Tensor from two 2d-Matrices by contraction
        Mc, Mt = self.tensors[c_qbit], self.tensors[t_qbit]
        T1 = np.einsum("ijk , lkm -> ijlm", Mc, Mt)

        ## Applying the 4d-gate on the 4d-tensor
        gate = MPS.MultiGates[op]
        T2 = np.einsum("klij, ijmn -> klmn", gate, T1)

        ## TODO: Check correctness
        ## TODO: Include bond_dim
        U, S, M2 = np.linalg.svd(T2)
        S = S * np.eye(2)
        M1 = np.einsum("ijkl, klm, ijl", U, S)
        self.tensors[c_qbit], self.tensors[t_qbit] = M1, M2

        ## Keep record & End of method
        self.gates_applied.append((c_qbit, t_qbit, op))

    def getAmplitude(self, qbit):
        ## TODO: Get the amplitude of a specific qubit
        pass

    def TEBD(self, op: str, *qubits):
        """
        Time-Evolution Block-Decimation (TEBD) Algorithm

        ++ Controlled application of Single/Multi Gates to an MPS
            * Single qubit gate is exact.
            * Multi qubit gate incurs small error controlled by the bond_dim.

        ++ References:
            * https://en.wikipedia.org/wiki/Time-evolving_block_decimation
            * https://www.youtube.com/watch?v=fq3_7vBcj3g&list=PLwNflrkTO97K1Uj1W09W84mz-U7fWy2VW&index=4&t=2103s
             from the minute 57:00 onwards

            * A Practical Introduction to Tensor Networks, Sec. 7.2
            * Efficient simulation of one-dimensional quantum many-body systems
            * Efficient classical simulation of slightly entangled quantum computations
        """
        assert 2 >= len(qubits) > 0
        assert op in MPS.SingleGates.keys() or op in MPS.MultiGates.keys()

        ## TODO: Track Fidelity
        ## TODO: ?? Extend method to be applicable to certain time steps  ??

        ## Single Qubit Gates
        if op in MPS.SingleGates.keys():
            self.applySingleGate(qbit=qubits[0], op=op)

        ## Controlled Qubit Gates
        if op in MPS.MultiGates.keys():
            self.applyMultiGate(c_qbit=qubits[0], t_qbit=qubits[1], op=op)

        ## TODO: Rotation Gates