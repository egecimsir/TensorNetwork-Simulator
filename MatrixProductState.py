import numpy as np


def isUnitary(arr: iter) -> bool:
    pass


def createRotationalUnitary(op: str, theta: float):
    assert op[-1] in ["X", "Y", "Z"], "Invalid operation."
    axis = op[-1]

    sin = np.sin
    cos = np.cos
    exp = np.exp
    gate: np.ndarray = ...

    if axis == "X":
        gate = np.array([[cos(theta/2), -1j*sin(theta/2)],
                         [-1j*sin(theta/2), cos(theta/2)]], dtype=complex)
    if axis == "Y":
        gate = np.array([[cos(theta / 2), -sin(theta / 2)],
                         [sin(theta / 2), cos(theta / 2)]], dtype=complex)
    if axis == "Z":
        gate = np.array([[exp(-1j * theta / 2), 0],
                         [0, exp(1j * theta / 2)]], dtype=complex)

    return gate


class MPS:
    """
    Matrix Product State Representation of a Quantum Circuit
    """

    Operations = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
    ControlledOps = ["C" + op for op in Operations]

    BasicGates = {
        "X": np.array([[0, 1],
                      [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j],
                      [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0],
                      [0, -1]], dtype=complex),
        "H": np.array([[1, 1],
                      [1, -1]], dtype=complex) / 2 ** (1 / 2),
        "SWAP": np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]]).reshape(2, 2, 2, 2)
    }

    event = {
        "time_step": None,
        "gate": None,
        "params": None,
        "t_qubit": None,
        "c_Qubit": None
    }

    @classmethod
    def qubit(cls, state: int):
        """
        Creates Qubits of the given basis state
        """
        assert state in [0, 1]
        tensor = np.eye(2, dtype=complex)[state]
        return tensor.reshape(2, 1, 1)

    @classmethod
    def createUnitary(cls, op: str, param=None) -> np.ndarray:
        """
        Creates a unitary gate with given operation name and parameters.
        """
        assert op in cls.Operations + cls.ControlledOps

        gate_unitary: np.ndarray
        controlled: bool = op in cls.ControlledOps
        parametrized: bool = param is not None

        if controlled:
            gate_unitary = np.eye(4, dtype=complex)
            if parametrized:
                gate_unitary[:2, :2] = createRotationalUnitary(op=op, theta=param)
            else:
                U = MPS.BasicGates[op]
                gate_unitary[:2, :2] = U
            gate_unitary.reshape(2, 2, 2, 2)

        else:  # SingleGate
            if parametrized:
                gate_unitary = createRotationalUnitary(op=op, theta=param)
            else:
                gate_unitary = MPS.BasicGates[op]

        return gate_unitary


    def __init__(self, num_qubits: int, state=None):
        assert num_qubits > 0
        self.n_qubits: int = num_qubits

        ## Initializing Tensors
        self.tensors: [np.ndarray] = [MPS.qubit(0) for _ in range(num_qubits)]
        self[0].reshape(2, 1)
        self[-1].reshape(2, 1)
        if state is not None:
            self.initialize(state)

        ## TODO: Track bond_dims of qubits
        self.time_step = 0
        self.history: [MPS.event] = []

    def __getitem__(self, item: int):
        assert abs(item) in range(self.n_qubits)
        return self.tensors[item]

    def __setitem__(self, item: int, value):
        assert abs(item) in range(self.n_qubits)
        self.tensors[item] = value

    def __iter__(self):
        pass

    def __next__(self):
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

    def initialize(self, arr: [bool]):
        """
        Initialize the circuit with a given bit string/list
        """
        assert len(arr) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, arr))), "Initial states must be 0 or 1"

        arr = list(map(int, arr))
        self.tensors = [MPS.qubit(arr[i]) for i in range(self.n_qubits)]

    def assignQubits(self, arr: [bool]):
        """
        Assigning classical state values {0, 1} to each qubit of the circuit
        """
        assert len(arr) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, arr))), "States must be 0 or 1"

        arr = list(map(int, arr))
        for i, state in enumerate(arr):
            self[i] = self[i][state]

    ## TODO:
    def getAmplitudeOf(self, state):
        """Get the amplitude of a specific qubit by using np.einsum"""
        assert len(state) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, state))), "States must be 0 or 1"
        pass

    def applyGate(self, gate: np.ndarray, qbit: int):
        assert gate.ndim == 2 and gate.shape == (2, 2)
        assert qbit in range(self.n_qubits)

        tensor = self.tensors[qbit]
        self[qbit] = np.einsum("lk, kij -> lij", gate, tensor)

    def applyControlled(self, gate: np.ndarray, c_qbit: int, t_qbit: int):
        assert gate.ndim == 4 and gate.shape == (2, 2, 2, 2)
        assert c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)
        assert t_qbit - c_qbit == 1

        ## Creating 4d-Tensor from two 2d-Matrices by contraction
        Mc, Mt = self[c_qbit], self[t_qbit]
        T1 = np.einsum("ijk , lkm -> ijlm", Mc, Mt)

        ## Applying the 4d-gate on the 4d-tensor
        T2 = np.einsum("klij, ijmn -> klmn", gate, T1)

        ## Singular Value Decomposition
        U, S, M2 = np.linalg.svd(T2)
        S = S * np.eye(2)
        M1 = np.einsum("ijkl, klm, ijl", U, S)

        ## Assign results back to qubits
        self[c_qbit], self[t_qbit] = M1, M2

    def SWAP(self, q1: int, q2: int):
        """
        Swaps two sequential qubits.
        """
        assert q1 in range(self.n_qubits) and q2 in range(self.n_qubits)
        assert abs(q1 - q2) == 1

        temp = self[q1]
        self[q1] = self[q2]
        self[q2] = temp

    def TEBD(self, op: str, param=None, *qubits):
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

        ## TODO: Track Fidelity
        """
        assert len(qubits) == 1 or len(qubits) == 2
        assert op in MPS.Operations + MPS.ControlledOps
        if param is not None:
            assert "R" in op

        ## Create the desired unitary matrix
        gate: np.ndarray = MPS.createUnitary(op=op, param=param)

        ## Apply the unitary gates using tensor contraction
        if len(qubits) == 1:
            self.applyGate(gate=gate, qbit=qubits[0])

        if len(qubits) == 2:
            self.applyControlled(gate=gate, c_qbit=qubits[0], t_qbit=qubits[1])