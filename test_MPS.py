import unittest
from MatrixProductState import *


class TestMPS(unittest.TestCase):

    def setUp(self):
        ## MPS with different n_qubits
        self.qc1 = MPS(1)
        self.qc2 = MPS(2)
        self.qc3 = MPS(3)
        self.qc5 = MPS(5)
        self.qc7 = MPS(7)
        self.qc21 = MPS(21)
        self.qc31 = MPS(31)

        ## All available gate operations
        self.operations = MPS.availableOps()

        ## Test qc.time_step
        ## Test qc.history

    def tearDown(self):
        ## TODO: Implement
        pass

    ######################
    #### Test Methods ####
    ######################

    def test_isUnitary(self):
        ## TODO: Implement
        pass

    def test_createRotationalUnitary(self):
        ## TODO: Implement
        pass

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

    def test_assignQubits(self):
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
        self.assertEqual(self.qc1.TEBD(...), ..., ...)
        pass

    def test_getAmplitudeOfState(self):
        ## TODO: Implement
        self.assertEqual(self.qc1.getAmplitudeOfState(...), ..., ...)
        pass


if __name__ == '__main__':
    unittest.main()
