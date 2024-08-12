import numpy as np


def is_unitary(arr: np.ndarray) -> bool:
    """
    Checks if a tensor is unitary
    """
    if not arr.dtype == complex:
        raise ValueError('Input tensor must be complex.')

    eye = np.eye(len(arr), dtype=complex)
    arr_t = arr.T.conj()
    return np.allclose(eye, arr.dot(arr_t))


def create_rotational_unitary(op: str, theta: float) -> np.ndarray:
    """
    Creates RX, RY, RZ gates for a given string={'X','Y','Z'}
    """
    if op not in ("X", "Y", "Z"):
        raise ValueError("Invalid operation")

    sin, cos, exp = np.sin, np.cos, np.exp
    gate: np.ndarray = ...

    if op == "X":
        gate = np.array([[cos(theta / 2), -1j * sin(theta / 2)],
                         [-1j * sin(theta / 2), cos(theta / 2)]], dtype=complex)
    if op == "Y":
        gate = np.array([[cos(theta / 2), -sin(theta / 2)],
                         [sin(theta / 2), cos(theta / 2)]], dtype=complex)
    if op == "Z":
        gate = np.array([[exp(-1j * theta / 2), 0],
                         [0, exp(1j * theta / 2)]], dtype=complex)

    return gate
