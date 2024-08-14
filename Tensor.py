import numpy as np
from utils import create_rotational_unitary


## Base Quantum Gates
BaseQuantumGates = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / 2 ** (1 / 2),
}


class Tensor:

    @classmethod
    def qubit(cls, state: int):
        assert int(state) in (0, 1)
        return cls(np.eye(2, dtype=complex)[state], name="Qubit")

    @classmethod
    def gate(cls, op: str, param=None):
        assert op in ("X", "Y", "Z", "H")
        if param is None:
            arr = BaseQuantumGates[op]
            name = op
        else:
            assert op != "H"
            arr = create_rotational_unitary(op, param)
            name = "R" + op
        return cls(arr, name)

    @classmethod
    def c_gate(cls, op: str, param=None):
        c_gate = np.eye(4, dtype=complex)
        gate = Tensor.gate(op, param)
        c_gate[2:, 2:] = gate.array
        return cls(c_gate, name="C" + gate.name).reshape(2, 2, 2, 2)

    def __init__(self, data, name=None):
        self.array = np.asarray(data, complex)
        self.name = name

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"Tensor{self.array.shape}"

    def __str__(self):
        return str(self.array)

    def __array__(self, dtype=complex):
        """Enables interoperability with numpy"""
        return np.asarray(self.array, dtype=dtype)

    @property
    def shape(self):
        return self.array.shape

    @property
    def bond_dim(self):
        return self.shape[-1]

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
