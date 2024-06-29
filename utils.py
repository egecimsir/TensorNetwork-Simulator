import numpy as np


def is_unitary(arr: np.ndarray) -> bool:
    """
    Checks if a tensor is unitary
    """
    if not arr.dtype == complex:
        raise TypeError('Input tensor must be complex.')

    eye = np.eye(len(arr), dtype=complex)
    arr_t = arr.T.conj()
    return np.allclose(eye, arr.dot(arr_t))


def make_qubit(arr) -> np.ndarray:
    state = np.asarray(arr, dtype=complex)
    qubit = state / np.linalg.norm(state)
    if not (is_unitary(state) and len(state) == 2):
        raise ValueError
    else:
        return qubit


def create_rotational_unitary(op: str, theta: float) -> np.ndarray:
    """
    Creates RX, RY, RZ gates for a given string={'X','Y','Z'}
    """
    if op[-1] not in ["X", "Y", "Z"]:
        raise ValueError("Invalid operation.")
    axis = op[-1]
    sin = np.sin
    cos = np.cos
    exp = np.exp
    gate: np.ndarray = ...

    if axis == "X":
        gate = np.array([[cos(theta / 2), -1j * sin(theta / 2)],
                         [-1j * sin(theta / 2), cos(theta / 2)]], dtype=complex)
    if axis == "Y":
        gate = np.array([[cos(theta / 2), -sin(theta / 2)],
                         [sin(theta / 2), cos(theta / 2)]], dtype=complex)
    if axis == "Z":
        gate = np.array([[exp(-1j * theta / 2), 0],
                         [0, exp(1j * theta / 2)]], dtype=complex)
    return gate


def to_int_list(state) -> [int]:
    return [int(i) for i in list(state)]