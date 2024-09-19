import numpy as np
import random
from datetime import datetime
from typing import Optional

from TensorNetwork import TensorNetwork as TN
from MatrixProductState import MPS
from QFTMPO import QFTMPO
from utils import runtime


@runtime
def prepare_mps(N: int, ent_level: Optional[float] = 0.0, max_bond: Optional[int] = None) -> MPS:
    return TN.generate_entangled_mps(n_qubits=N, ent_level=ent_level, bond_dim=max_bond)


@runtime
def prepare_mpo(N: int, max_bond: Optional[int] = None) -> QFTMPO:
    return QFTMPO(n_qubits=N).zip_up(bond_dim=max_bond)


@runtime
def prepare_entangled_state(N: int, depth: int = 1, ent_level: Optional[float] = 1.0) -> MPS:
    assert 0 <= ent_level <= 1
    assert depth >= 1

    qc = TN(n_qubits=N)
    qubits = int(N * ent_level)

    for _ in range(depth):
        ## Entangle each two qubits up to given percentage of qubits
        for i in range(0, qubits, 2):
            qc.x(i, param=2 * np.pi / random.randint(1, 48))
            if i == N - 1:
                break
            qc.c_not(i, i + 1)

        ## Entangle all qubits that are already entangled
        for i in range(1, qubits, 2):
            qc.x(i, param=2 * np.pi / random.randint(1, 48))
            if i == N - 1:
                break
            qc.c_not(i, i + 1)

    ## Entangle first and last qubit
    qc.move_tensor(T=N-1, to=1)
    qc.c_not(0, 1)
    qc.move_tensor(T=1, to=N-1)

    return qc.mps


@runtime
def prepare_AME_state(N: int) -> MPS:
    """
    Absoulately Maximally Entangled State
    -------------------------------------
    DOI: 10.1103/PhysRevA.100.022342
    """
    qc = TN(n_qubits=N)

    for i in range(N):
        qc.hadamard(i)

    for i in range(N-2):
        qc.c_not(i, i+1)

    qc.move_tensor(T=N - 1, to=1)
    qc.c_not(0, 1)
    qc.move_tensor(T=1, to=N - 1)

    return qc.mps


@runtime
def qft_with_gates(mps: MPS, max_bond: Optional[int] = None) -> MPS:

    qc = TN.from_MPS(mps=mps, bond_dim=max_bond)

    ## QFT
    for i in range(len(mps)):
        qc.hadamard(i)
        for j in range(i + 1, len(mps)):
            if i + 1 != j:  qc.move_tensor(T=j, to=i + 1)
            qc.c_phase(i, i + 1, phase=np.pi / 2 ** (j - i))
            if i + 1 != j:  qc.move_tensor(T=i + 1, to=j)

    return qc.mps


@runtime
def qft_with_mpo(mps: MPS, max_bond: Optional[int] = None) -> MPS:
    mpo, _ = prepare_mpo(N=len(mps), max_bond=max_bond)
    return mpo(mps=mps)

