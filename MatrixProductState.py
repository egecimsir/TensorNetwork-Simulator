import numpy as np
from Utils import *
from Exceptions import *


class TensorNetworks:
    """
    Base class for Matrix Product States
    """
    name: str = "???"
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
    ops: list = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
    controlled_ops: list = ["C" + op for op in ops]

    @classmethod
    def available_ops(cls, _print_=False) -> list:
        allOps = cls.ops + cls.controlled_ops
        allOps.append("SWAP")
        if _print_:
            print(f"Available Operations: {allOps}")
        return allOps

    @classmethod
    def qubit(cls, state: int) -> np.ndarray:
        """
        Creates Qubits of the given basis state
        """
        if state not in [0, 1]:
            raise ValueError("Qubit state must be 0 or 1")
        return np.eye(2, dtype=complex)[state].reshape(2, 1, 1)

    @classmethod
    def create_unitary(cls, op: str, param=None) -> np.ndarray:
        """
        Creates a unitary gate with given operation name and parameters.
        """
        if op not in cls.ops + cls.controlled_ops:
            raise InvalidOperation

        gate_unitary: np.ndarray
        controlled: bool = op in cls.controlled_ops
        parametrized: bool = param is not None

        if controlled:
            gate_unitary = np.eye(4, dtype=complex)
            if parametrized:
                gate_unitary[:2, :2] = createRotationalUnitary(op=op, theta=param)
            else:
                U = cls.BaseQuantumGates[op]
                gate_unitary[:2, :2] = U
            gate_unitary.reshape(2, 2, 2, 2)

        else:  # SingleGate
            if parametrized:
                gate_unitary = createRotationalUnitary(op=op, theta=param)
            else:
                gate_unitary = cls.BaseQuantumGates[op]

        return gate_unitary

    def __init__(self, num_qubits, name=name):
        self.name = name
        self.print_OUT = True

        self.n_qubits: int = num_qubits
        self.tensors: [np.ndarray] = []
        self.basis_states: [np.ndarray] = self.get_basis_states()

        ## Tracking
        self.time_step = 0
        self.event_log = []
        self.index = 0

    def __getitem__(self, item: int):
        if abs(item) not in range(self.n_qubits):
            raise IndexError
        return self.tensors[item]

    def __setitem__(self, item: int, value):
        if abs(item) not in range(self.n_qubits):
            raise IndexError
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
        st = f"TensorNetwork({self.n_qubits})\n"
        for q in self.tensors:
            __ = f"\n{str(q)}\n\n"
            st = st + __
        return st

    def __str__(self):
        return "\n\n".join([str(q) for q in self.tensors])

    def valid_state(self, state) -> bool:
        state = to_int_list(state)
        qubits_match: bool = len(state) == self.n_qubits
        all_qubits_binary: bool = np.all(list(map(lambda x: x == 0 or x == 1, state)))

        return qubits_match and all_qubits_binary

    def get_basis_states(self) -> tuple:
        return tuple([bin(i)[2:].zfill(self.n_qubits) for i in range(2**self.n_qubits)])

    def execution_time(func: callable) -> callable:
        from datetime import datetime
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            ## Calculate execution time (delta)
            begin = datetime.now()
            event: dict = func(self, *args, **kwargs)
            end = datetime.now()
            delta = (end - begin).microseconds

            ## Update objects last event
            if func.__name__ == "TEBD":
                event["exec_time"] = delta

            ## Print out
            if self.print_OUT:
                print(f"\n{func.__name__} Execution time: {delta} microseconds\n")

        return wrapper

    def apply_gate(self, gate_U: np.ndarray, qbit: int):
        """
        Performs the matrix multiplication: gate_U * tensor
        """
        if not (gate_U.ndim == 2 and gate_U.shape == (2, 2)):
            raise InvalidGate
        if not isUnitary(gate_U):
            raise InvalidGate
        if qbit not in range(self.n_qubits):
            raise IndexError

        ## Tensor contraction
        tensor = self[qbit]
        self[qbit] = np.einsum("lk, kij -> lij", gate_U, tensor)

    def apply_controlled(self, gate_U: np.ndarray, c_qbit: int, t_qbit: int):
        """
        Performs the 4d-tensor contraction between qubits and the gate
        """
        if not (gate_U.ndim == 4 and gate_U.shape == (2, 2, 2, 2)):
            raise InvalidGate
        if not (c_qbit in range(self.n_qubits) and t_qbit in range(self.n_qubits)):
            raise IndexError
        if (t_qbit - c_qbit) != 1:
            raise InvalidOperation

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


## TODO: Track bond_dims of qubits
## TODO: Improve exceptions
class MatrixProductState(TensorNetworks):
    """
    Matrix Product State Representation of a Quantum Circuit
    """
    name = "MatrixProductState"

    def __init__(self, num_qubits: int, state=None, name=name):
        super().__init__(num_qubits, name)
        if state is None:
            self.initialize([0 for _ in range(self.n_qubits)])
        else:
            self.initialize(state)

    def check_shapes(self) -> bool:
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

    def initialize(self, state):
        """
        Initialize the circuit with a given bit string/list
        """
        if not self.valid_state(state):
            raise InitializationError

        arr = to_int_list(state)
        for i in range(self.n_qubits):
            qubit = TensorNetworks.qubit(arr[i])
            if i == 0 or i == self.n_qubits - 1:
                ## Edge qubits must be rank-2, others rank-3
                qubit = qubit[:, :, 0]
            self.tensors.append(qubit)

        if not self.check_shapes():
            raise InitializationError

    def get_amplitude_of(self, state: str) -> float:
        """
        Get the amplitude of a specific qubit by using np.einsum
        """
        if not self.valid_state(state):
            raise InitializationError

        arr = to_int_list(state)
        ## Set the basis-state indices of qubits
        for i, state in enumerate(arr):
            self[i] = self[i][state]

        ## Contract middle qubits
        res = np.eye(2, dtype=complex)
        for qubit in self.tensors[1: self.n_qubits - 1]:
            res = np.matmul(res, qubit)

        ## Contract edge qubits
        res = res @ self.tensors[-1]
        res = self.tensors[0] @ res

        return res

    def get_probabilities(self):
        ## TODO: Insert checks
        amplitudes = []
        for state in self.basis_states:
            amplitudes.append(self.get_amplitude_of(state))
        amplitudes = np.array(amplitudes, dtype=complex)
        return amplitudes * amplitudes.T.conj()

    def SWAP(self, q1: int, q2: int):
        """
        Swaps two sequential qubits.
        """
        if not (q1 in range(self.n_qubits) and q2 in range(self.n_qubits)):
            raise IndexError
        if abs(q1 - q2) != 1:
            raise InvalidOperation("Swapped Qubits must be sequential!")

        ## Swap Qubits
        temp = self[q1]
        self[q1] = self[q2]
        self[q2] = temp

        ## Create event
        event = dict(time_step=self.time_step + 1,
                     op="SWAP",
                     param=None,
                     unitary=None,
                     c_qubit=q1,
                     t_qubit=q2)

        ## Increase time step
        self.event_log.append(event)
        self.time_step += 1

        return event

    @TensorNetworks.execution_time
    def TEBD(self, op: str, qubits, param=None):
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

        ## Check input conditions
        if type(qubits) is list:
            if not (len(qubits) == 1 or len(qubits) == 2):
                raise ValueError("Only one or two qubit operations are supported")
        elif type(qubits) is int:
            if qubits not in range(self.n_qubits):
                raise IndexError
            qubits = [qubits]
        else:
            raise InvalidOperation

        if op not in TensorNetworks.available_ops():
            raise InvalidOperation("Operation not available!")
        if param is not None and "R" not in op:
            raise InvalidOperation("Only rotational gates (R_, CR_) can have parameters")

        ## Create an event for recording
        event: dict = {}
        ## Create the desired unitary matrix
        gate: np.ndarray = TensorNetworks.create_unitary(op=op, param=param)

        ## Apply Single Qubit Gate
        if len(qubits) == 1:
            self.apply_gate(gate_U=gate, qbit=qubits[0])
            event = dict(time_step=self.time_step + 1,
                         op=op,
                         param=param,
                         unitary=gate,
                         c_qubit=None,
                         t_qubit=qubits[0],
                         exec_time=None)
        ## Apply Multi Qubit Gate
        elif len(qubits) == 2:
            self.apply_controlled(gate_U=gate, c_qbit=qubits[0], t_qbit=qubits[1])
            event = dict(time_step=self.time_step + 1,
                         op=op,
                         param=param,
                         unitary=gate,
                         c_qubit=qubits[0],
                         t_qubit=qubits[1],
                         exec_time=None)

        ## Update history with current event
        self.event_log.append(event)
        self.time_step += 1

        return event
