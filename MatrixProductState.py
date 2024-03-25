import numpy as np
from Utils import createRotationalUnitary
from Exceptions import *


class MPS:
    """
    Matrix Product State Representation of a Quantum Circuit
    """
    ## TODO: Track bond_dims of qubits
    ## TODO: Replace assertions with custom exceptions

    ops: list = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
    controlled_ops: list = ["C" + op for op in ops]

    BaseQuantumGates = {
        "X": np.array([[0, 1],
                       [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j],
                       [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0],
                       [0, -1]], dtype=complex),
        "H": np.array([[1, 1],
                       [1, -1]], dtype=complex) / 2 ** (1 / 2),
    }

    @classmethod
    def availableOps(cls) -> list:
        return cls.ops + cls.controlled_ops

    @classmethod
    def qubit(cls, state: int) -> np.ndarray:
        """
        Creates Qubits of the given basis state
        """
        if state not in [0, 1]:
            raise ValueError("Qubit state must be 0 or 1")
        return np.eye(2, dtype=complex)[state].reshape(2, 1, 1)

    @classmethod
    def createUnitary(cls, op: str, param=None) -> np.ndarray:
        """
        Creates a unitary gate with given operation name and parameters.
        """
        if op not in cls.ops + cls.controlled_ops:
            raise InvalidOperationException("")

        gate_unitary: np.ndarray
        controlled: bool = op in cls.controlled_ops
        parametrized: bool = param is not None

        if controlled:
            gate_unitary = np.eye(4, dtype=complex)
            if parametrized:
                gate_unitary[:2, :2] = createRotationalUnitary(op=op, theta=param)
            else:
                U = MPS.BaseQuantumGates[op]
                gate_unitary[:2, :2] = U
            gate_unitary.reshape(2, 2, 2, 2)

        else:  # SingleGate
            if parametrized:
                gate_unitary = createRotationalUnitary(op=op, theta=param)
            else:
                gate_unitary = MPS.BaseQuantumGates[op]

        return gate_unitary

    def __init__(self, num_qubits: int, state=None):
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive integer")
        self.n_qubits: int = num_qubits
        self.index = 0

        ## Initializing Tensors
        self.tensors: [np.ndarray] = []
        if state is None:
            self.initialize([0 for _ in range(num_qubits)])
        else:
            self.initialize(state)

        ## Tracking
        self.time_step = 0
        self.history = []

    def __getitem__(self, item: int):
        if abs(item) not in range(self.n_qubits):
            raise IndexError()
        return self.tensors[item]

    def __setitem__(self, item: int, value):
        if abs(item) not in range(self.n_qubits):
            raise IndexError()
        self.tensors[item] = value

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > self.n_qubits - 1:
            raise StopIteration
        else:
            self.index += 1
            return self.tensors[self.index - 1]

    def __repr__(self):
        st = f"QuantumCircuit({self.n_qubits})\n"
        for q in self.tensors:
            __ = f"\n{str(q)}\n\n"
            st = st + __
        return st

    def __str__(self):
        return "\n\n".join([str(q) for q in self.tensors])

    def initialize(self, arr: [bool]):
        """
        Initialize the circuit with a given bit string/list
        """
        if len(arr) != self.n_qubits:
            raise UnmatchedQubitsException
        if not np.all(list(map(lambda x: x == 0 or x == 1, arr))):
            raise ValueError("Initial states must be 0 or 1")
        arr = list(map(int, arr))

        for i in range(self.n_qubits):
            qubit = MPS.qubit(arr[i])
            if i == 0 or i == self.n_qubits - 1:
                ## Edge qubits must be rank-2, others rank-3
                qubit = qubit[:, :, 0]

            self.tensors.append(qubit)

    def checkShapes(self) -> bool:
        """
        Check if the tensors has the correct shape:
            ++ Boundary Qubits are rank-2
            ++ Middle Qubits are rank-3
        """
        results: [bool] = []
        for i, tensor in enumerate(self.tensors):
            if i == 0 or i == self.n_qubits - 1:
                res = tensor.ndim == 2 and tensor.dtype == complex and tensor.shape == (2, 1)
            else:
                res = tensor.ndim == 3 and tensor.dtype == complex and tensor.shape == (2, 1, 1)
            results.append(res)

        return np.all(results)

    def assignQubits(self, arr: [bool]):
        """
        Assigning classical state values {0, 1} to each qubit of the circuit
        """
        if len(arr) != self.n_qubits:
            raise UnmatchedQubitsException
        if not np.all(list(map(lambda x: x == 0 or x == 1, arr))):
            raise ValueError("States must be 0 or 1")
        arr = list(map(int, arr))

        for i, state in enumerate(arr):
            self[i] = self[i][state]

    def getAmplitudeOfState(self, state):
        """
        Get the amplitude of a specific qubit by using np.einsum
        """
        if len(state) != self.n_qubits:
            raise UnmatchedQubitsException
        if not np.all(list(map(lambda x: x == 0 or x == 1, state))):
            raise ValueError("States must be 0 or 1")

        ## Set the basis-state indices of qubits
        self.assignQubits(state)

        ## Contract  qubits
        res = np.eye(2, dtype=complex)
        for qubit in self.tensors[1: self.n_qubits - 1]:
            res = np.matmul(res, qubit)

        res = res @ self.tensors[-1]
        res = self.tensors[0] @ res

        return res

    def applyGate(self, gate_U: np.ndarray, qbit: int):
        if not (gate_U.ndim == 2 and gate_U.shape == (2, 2)):
            raise InvalidGateException
        if qbit not in range(self.n_qubits):
            raise IndexError

        ## Tensor contraction
        tensor = self[qbit]
        self[qbit] = np.einsum("lk, kij -> lij", gate_U, tensor)

    def applyControlled(self, gate_U: np.ndarray, c_qbit: int, t_qbit: int):
        if not (gate_U.ndim == 4 and gate_U.shape == (2, 2, 2, 2)):
            raise InvalidGateException
        if (c_qbit not in range(self.n_qubits)) or (t_qbit not in range(self.n_qubits)):
            raise IndexError
        if (t_qbit - c_qbit) != 1:
            raise UnmatchedQubitsException

        ## Creating 4d-Tensor from two 2d-Matrices by contraction
        Mc, Mt = self[c_qbit], self[t_qbit]
        T1 = np.einsum("ijk , lkm -> ijlm", Mc, Mt)

        ## Applying the 4d-gate on the 4d-tensor
        T2 = np.einsum("klij, ijmn -> klmn", gate_U, T1)

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
        if q1 not in range(self.n_qubits) or q2 not in range(self.n_qubits):
            raise IndexError
        if abs(q1 - q2) != 1:
            raise UnmatchedQubitsException

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
        if not (len(qubits) == 1 or len(qubits) == 2):
            raise ValueError("Only one or two qubit operations are supported")
        if op not in MPS.availableOps():
            raise InvalidOperationException
        if param is not None:
            if "R" not in op:
                ## "Rotational Gates (R_, CR_) can have parameters"
                raise InvalidOperationException
        event = {}

        ## Create the desired unitary matrix
        gate: np.ndarray = MPS.createUnitary(op=op, param=param)

        ## Apply the unitary gates & Create events
        if len(qubits) == 1:
            self.applyGate(gate_U=gate, qbit=qubits[0])
            event = dict(time_step=self.time_step + 1, op=op, param=param,
                         unitary=gate, c_qubit=None, t_qubit=qubits[0])

        elif len(qubits) == 2:
            self.applyControlled(gate_U=gate, c_qbit=qubits[0], t_qbit=qubits[1])
            event = dict(time_step=self.time_step + 1, op=op, param=param,
                         unitary=gate, c_qubit=qubits[0], t_qubit=qubits[1])

        ## Update history with current event
        self.history.append(event)
        self.time_step += 1
