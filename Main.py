import numpy as np
from MatrixProductState import MatrixProductState

if __name__ == "__main__":

    mps = MatrixProductState(3)
    mps.TEBD("H",  1)
    print(mps)

    event = mps.event_log[-1]
    print(event)
