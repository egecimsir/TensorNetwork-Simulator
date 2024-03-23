import numpy as np
from Tensor import Tensor
from Utils import createRotational


class MPS:
    """
    Tensor Representation of Quantum Circuit
    """
    BasicGates = {
        "X": Tensor([[0, 1], [1, 0]]),
        "Y": Tensor([[0, -1j], [1j, 0]]),
        "Z": Tensor([[1, 0], [0, -1]]),
        "H": Tensor([[1, 1], [1, -1]]) / 2 ** (1 / 2),
        "SWAP": Tensor([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]]).reshape(2, 2, 2, 2)
    }

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
        self.tensors = [Tensor.qubit(arr[i]) for i in range(self.n_qubits)]

    def setState(self, arr: [bool]):
        """
        Assigning classical state values {0, 1} to each qubit of the circuit
        """
        assert len(arr) == self.n_qubits, "Number of qubits does not match"
        assert np.all(list(map(lambda x: x == 0 or x == 1, arr))), "States must be 0 or 1"

        arr = list(map(int, arr))
        for i, state in enumerate(arr):
            self.tensors[i] = self.tensors[i][state]

    def getAmplitudeOf(self, qbit):
        ## TODO: Get the amplitude of a specific qubit by using np.einsum
        pass

    @classmethod
    def createUnitary(cls, op: str, param=None) -> np.ndarray:
        """
        Creates a unitary gate with given operation name and parameters.
        """
        assert op in cls.BasicGates.keys()

        gate: np.ndarray
        controlled: bool = op[0] == "C"
        parametrized: bool = param is not None

        if controlled:
            gate = np.eye(4, dtype=complex)
            if parametrized:
                gate[:2, :2] = createRotational(axis=op, theta=param)
            else:
                U = MPS.BasicGates[op]
                gate[:2, :2] = U

            gate.reshape(2, 2, 2, 2)

        else:  # SingleGate
            if parametrized:
                gate = createRotational(axis=op, theta=param)
            else:
                gate = MPS.BasicGates[op]
            assert gate.shape == (2, 2)

        return gate

    def applyGate(self, gate: np.ndarray, qubit: int):
        assert gate.ndim == 2 and gate.shape == (2, 2)
        assert qubit in range(self.n_qubits)

    def applyControlled(self, gate: np.ndarray, c_qbit: int, t_qbit: int):
        assert gate.ndim == 4 and gate.shape == (2, 2, 2, 2)
        assert c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)
        assert t_qbit - c_qbit == 1

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

        gate = MPS.createUnitary(op=op, param=param)

        if len(qubits) == 1:
            self.applyGate(gate=gate, qubit=qubits[0])

        if len(qubits) == 2:
            self.applyControlled(gate=gate, c_qbit=qubits[0], t_qbit=qubits[1])
