import numpy as np


def isUnitary(arr: iter) -> bool:
    pass


def createRotational(axis: str, theta: float):
    assert axis in ["X", "Y", "Z"], "Unrecognized operation"
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
