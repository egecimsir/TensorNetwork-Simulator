import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Tuple, List

from Tensor import Tensor
from TensorNetwork import TensorNetwork as TN
from MatrixProductState import MPS
from QFTMPO import QFTMPO


def runtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        runtime = end-start
        return result, runtime

    return wrapper


@runtime
def prepare_mps(N: int, ent_level: Optional[float] = 0.0, bond_dim: Optional[int] = None) -> MPS:
    return TN.generate_entangled_circuit(n_qubits=N, ent_level=ent_level, bond_dim=bond_dim)


mps, runtime = prepare_mps(10, 0.5)
print(runtime)
