import numpy as np


def isUnitary(arr: iter) -> bool:
    pass


def createRotationalUnitary(op: str, theta: float):
    assert op[-1] in ["X", "Y", "Z"], "Invalid operation."
    axis = op[-1]

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
