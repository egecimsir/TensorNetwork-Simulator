import unittest
import numpy as np

from utils import *


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        ## TODO: Create unitary arrays
        self.arr1 = np.array([1, 2, 3, 4, 5])
        self.arr2 = ...
        self.arr3 = ...

        ## TODO: Create non-unitary arrays
        self.arr4 = ...
        self.arr5 = ...
        self.arr6 = ...

        ## All available operations of MPS Class
        ops = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
        c_ops = ["C" + op for op in ops]
        all_ops = ops + c_ops
        self.ops = all_ops


    def test_isUnitary(self):
        ## Check unitary
        self.assertTrue(is_unitary(self.arr1))
        self.assertTrue(is_unitary(self.arr2))
        self.assertTrue(is_unitary(self.arr3))

        ## Check non-unitary
        self.assertFalse(is_unitary(self.arr4))
        self.assertFalse(is_unitary(self.arr5))
        self.assertFalse(is_unitary(self.arr6))

        pass

    def test_createRotationalUnitary(self):
        """
        Creates RX, RY, RZ gates for a given string={'X','Y','Z'}
        """
        ## TODO: Create random seed for theta and adjust function
        theta = ...

        sin = np.sin
        cos = np.cos
        exp = np.exp
        X = np.array([[cos(theta / 2), -1j * sin(theta / 2)],
                      [-1j * sin(theta / 2), cos(theta / 2)]], dtype=complex)

        Y = np.array([[cos(theta / 2), -sin(theta / 2)],
                      [sin(theta / 2), cos(theta / 2)]], dtype=complex)

        Z = np.array([[exp(-1j * theta / 2), 0],
                      [0, exp(1j * theta / 2)]], dtype=complex)

        self.assertEqual(X, create_rotational_unitary(op="X", theta=theta))
        self.assertEqual(X, create_rotational_unitary(op="RX", theta=theta))

        self.assertEqual(Y, create_rotational_unitary(op="Y", theta=theta))
        self.assertEqual(Y, create_rotational_unitary(op="RY", theta=theta))

        self.assertEqual(Z, create_rotational_unitary(op="Z", theta=theta))
        self.assertEqual(Z, create_rotational_unitary(op="RZ", theta=theta))

        pass
