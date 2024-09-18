import numpy as np
from datetime import datetime
from typing import Optional

from TensorNetwork import TensorNetwork as TN
from MatrixProductState import MPS
from QFTMPO import QFTMPO


def runtime(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        runtime = (end-start).total_seconds() * 10**3  ## ms
        return result, runtime
    return wrapper


@runtime
def prepare_mps(N: int, ent_level: Optional[float] = 0.0, max_bond: Optional[int] = None) -> MPS:
    return TN.generate_entangled_circuit(n_qubits=N, ent_level=ent_level, bond_dim=max_bond)


@runtime
def prepare_mpo(N: int, max_bond: Optional[int] = None) -> QFTMPO:
    return QFTMPO(n_qubits=N).zip_up(bond_dim=max_bond)


@runtime
def qft_with_gates(N: int, ent_level: Optional[float] = 0.0, max_bond: Optional[int] = None) -> MPS:
    mps, _ = prepare_mps(N, ent_level=ent_level, max_bond=max_bond)
    qc = TN.from_MPS(mps=mps)

    ## QFT
    for i in range(N):
        qc.hadamard(i)
        for j in range(i + 1, N):
            if i + 1 != j:  qc.move_tensor(T=j, to=i + 1)
            qc.c_phase(i, i + 1, phase=np.pi / 2 ** (j - i))
            if i + 1 != j:  qc.move_tensor(T=i + 1, to=j)

    return qc.mps


@runtime
def qft_with_mpo(N: int, ent_level: Optional[float] = 0.0, max_bond: Optional[int] = None) -> MPS:
    mps, _ = prepare_mps(N, ent_level=ent_level)
    mpo = prepare_mpo(N=N, max_bond=max_bond)
    return mpo(mps=mps)

