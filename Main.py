import numpy as np
from MatrixProductState import MatrixProductState

if __name__ == "__main__":

    mps = MatrixProductState(4)

    mps.TEBD("H", 1)
    mps.TEBD("CX", [1, 2])
    mps.print_out()
    print(mps.event_log[-1])


