import unittest
from MatrixProductState import MPS
from qiskit.circuit.library import QFT


## TODO: Create identical QFT circuits and compare their results
class CompareQFTCircuits(unittest.TestCase):
    def setUp(self):

        num_qubits = 5

        qc = QFT(num_qubits)
        mps = MPS(num_qubits)

        pass

