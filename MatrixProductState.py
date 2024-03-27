import numpy as np
from TensorNetworks import *


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

    def apply_controlled_gate(self, gate_U: np.ndarray, c_qbit: int, t_qbit: int):
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

    @TensorNetworks.execute
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

    @TensorNetworks.execute
    def get_probabilities(self) -> np.ndarray:
        ## TODO: Insert checks
        amplitudes = []
        for state in self.basis_states:
            amplitudes.append(self.get_amplitude_of(state))

        ## Array of all complex amplitudes
        amplitudes = np.array(amplitudes, dtype=complex)

        ## Probabilities of given states
        return amplitudes * amplitudes.T.conj()

    @TensorNetworks.execute
    def SWAP(self, q1: int, q2: int) -> dict:
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

    @TensorNetworks.execute
    def TEBD(self, op: str, qubits, param=None) -> dict:
        """
        Time-Evolution Block-Decimation (TEBD) Algorithm
        """
        ## TODO: Track Fidelity

        ## Check input conditions
        if type(qubits) is list:
            if not (len(qubits) == 1 or len(qubits) == 2):
                raise ValueError("Only one or two qubit operations are supported")
        elif type(qubits) is int:
            if qubits not in range(self.n_qubits):
                raise IndexError
            qubits = [qubits]
        else:
            raise TypeError("Input must be a list or integer!")
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
            self.apply_controlled_gate(gate_U=gate, c_qbit=qubits[0], t_qbit=qubits[1])
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
