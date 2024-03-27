import numpy as np
from MatrixProductState import MPS

if __name__ == "__main__":
    ##qc = MPS(5, (0,1,1,0,0))

    qc = MPS(3)
    print(qc)

    qc.TEBD("H", None, 1)
    print(qc)

    qc.get_amplitude_of("101")