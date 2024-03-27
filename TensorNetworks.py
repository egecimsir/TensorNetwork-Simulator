import numpy as np

from Exceptions import *
from Utils import *


class TensorNetworks:
    """
    Base class for Matrix Product States
    """
    name: str = "TensorNetwork"
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
        self.total_runtime = 0
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

    def print_out(self) -> str:
        print(self)
        return str(self)

    def valid_state(self, state) -> bool:
        state = to_int_list(state)
        qubits_match: bool = len(state) == self.n_qubits
        all_qubits_binary: bool = np.all(list(map(lambda x: x == 0 or x == 1, state)))

        return qubits_match and all_qubits_binary

    def get_basis_states(self) -> tuple:
        return tuple([bin(i)[2:].zfill(self.n_qubits) for i in range(2 ** self.n_qubits)])

    def execute(func: callable) -> callable:
        from datetime import datetime
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            ## Calculate execution time (delta)
            if func.__name__ in ("TEBD", "SWAP"):
                begin = datetime.now()
                event: dict = func(self, *args, **kwargs)
                end = datetime.now()
                delta = (end - begin).microseconds
                ## Update the event of the object
                event["exec_time"] = delta
            else:
                begin = datetime.now()
                func(self, *args, **kwargs)
                end = datetime.now()
                delta = (end - begin).microseconds

            self.total_runtime += delta

            ## Print out
            if self.print_OUT:
                print(f"\n{func.__name__} Execution time: {delta * 1000}ms\n")

        return wrapper
