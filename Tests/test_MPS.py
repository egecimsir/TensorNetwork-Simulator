import unittest
from MatrixProductState import *


class TestMPS(unittest.TestCase):

    def setUp(self):
        ## MPS with different n_qubits
        self.qc1 = MatrixProductState(1)
        self.qc2 = MatrixProductState(2)
        self.qc3 = MatrixProductState(3)
        self.qc5 = MatrixProductState(5)
        self.qc7 = MatrixProductState(7)
        self.qc21 = MatrixProductState(21)
        self.qc31 = MatrixProductState(31)

        ## All available gate operations
        self.operations = MatrixProductState.available_ops()

        ## Test qc.time_step
        ## Test qc.history

    ##############################
    ######## Test Methods ########
    ##############################

    def test_availableOps(self):
        ops = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
        c_ops = ["C" + op for op in ops]
        all_ops = ops + c_ops

        self.assertEqual(self.qc1.available_ops(), all_ops)

    def test_qubit(self):
        ## TODO: Implement
        pass

    def test_createUnitary(self):
        ## TODO: Implement
        pass

    def test_initialize(self):
        ## TODO: Implement
        pass

    def test_checkShapes(self):
        ## TODO: Implement
        pass

    def test_applyGate(self):
        ## TODO: Implement
        pass

    def test_applyControlled(self):
        ## TODO: Implement
        pass

    def test_SWAP(self):
        ## TODO: Implement
        pass

    def test_TEBD(self):
        ## TODO: Implement
        self.assertEqual(self.qc1.TEBD(...), ...)
        pass

    def test_getAmplitudeOfState(self):
        ## TODO: Implement
        self.assertEqual(self.qc1.get_amplitude_of(...), ...)
        pass


if __name__ == '__main__':
    unittest.main()
