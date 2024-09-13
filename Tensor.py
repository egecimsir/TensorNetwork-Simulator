import numpy as np

from utils import create_rotational_unitary
from typing import Optional


## Quantum Gates
BaseQuantumGates = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / 2 ** (1 / 2),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
}


class Tensor:
    """
    Imitation of the np.ndarray, with only complex values
    """
    @classmethod
    def qubit(cls, state: int):
        assert int(state) in (0, 1)
        return cls(np.eye(2, dtype=complex)[state], name="Qubit").reshape(1, 2, 1)

    @classmethod
    def gate(cls, op: str, param: Optional[float] = None):
        assert op in ("X", "Y", "Z", "H")
        if param is None:
            arr = BaseQuantumGates[op]
            name = op
        else:
            assert op != "H"
            arr = create_rotational_unitary(op, param)
            name = "R" + op + f"({param:.2f})"
        return cls(arr, name)

    @classmethod
    def c_gate(cls, op: str, param: Optional[float] = None):
        if op == "SWAP":
            c_gate = BaseQuantumGates[op]
            name = op
        else:
            gate = Tensor.gate(op, param)
            c_gate = np.eye(4, dtype=complex)
            c_gate[2:, 2:] = gate.array
            name = "C" + gate.name

        return cls(c_gate, name=name).reshape(2, 2, 2, 2)

    @classmethod
    def phase_tensor(cls, phase: float, ndim: int = 2):
        assert ndim in (2, 3, 4)
        arr = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, np.exp(np.exp(1j * phase))]])
        if ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if ndim == 4:
            arr = arr.reshape(2, 2, 2, 2)

        return cls(arr, name=f"P{ndim}({phase:.2f})")

    @classmethod
    def copy_tensor(cls):
        arr = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
        return cls(arr, name="Copy")

    def __init__(self, data, name: Optional[str] = "Tensor"):
        self.array = np.asarray(data, complex)
        self.name = name

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"{self.name}{self.array.shape}"

    def __str__(self):
        return str(self.array)

    def __array__(self, dtype=complex):
        """Enables interoperability with numpy"""
        return np.asarray(self.array, dtype=dtype)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def conj(self):
        return self.array.conjugate()

    def reshape(self, *shape):
        self.array = self.array.reshape(shape)
        return self

    def conjugate(self):
        self.array = self.array.conjugate()
        return self

    def squeeze(self, axis):
        self.array = self.array.squeeze(axis)
        return self
