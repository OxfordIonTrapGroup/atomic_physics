import unittest
from sympy.physics import wigner
from ion_phys.wigner import wigner3j
import numpy as np


class TestWigner(unittest.TestCase):
    def test_wigner(self):
        """ Cross-check our Wigner 3j symbol calculation against the sympy
        implementation.
        """
        j = [0, 0.5, 1., 1.5, 2., 2.5, 3]
        for j1 in j:
            for j2 in j:
                for j3 in j:
                    for m1 in np.arange(-j1, j1+1):
                        for m2 in np.arange(-j2, j2+1):
                            for m3 in np.arange(-j3, j3+1):
                                ip = wigner3j(j1, j2, j3, m1, m2, m3)
                                sp = wigner.wigner_3j(j1, j2, j3, m1, m2, m3)
                                self.assertTrue(np.abs(ip - sp) < 1e-15)


if __name__ == '__main__':
    unittest.main()
