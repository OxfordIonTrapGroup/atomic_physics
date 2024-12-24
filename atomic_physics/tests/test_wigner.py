import unittest

import numpy as np

import atomic_physics.wigner as ap_wigner

# from sympy.physics import wigner
from . import utils as sp_wigner  # HACK: waiting for new sympy release


class TestWigner(unittest.TestCase):
    def test_wigner(self):
        """Cross-check our Wigner 3j symbol calculation against the sympy
        implementation.
        """
        j = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3]
        for j1 in j:
            for j2 in j:
                for j3 in j:
                    for m1 in np.arange(-j1, j1 + 1):
                        for m2 in np.arange(-j2, j2 + 1):
                            for m3 in np.arange(-j3, j3 + 1):
                                wigner_ap = ap_wigner.wigner3j(j1, j2, j3, m1, m2, m3)
                                wigner_sp = sp_wigner.wigner_3j(j1, j2, j3, m1, m2, m3)
                                self.assertTrue(np.abs(wigner_ap - wigner_sp) < 1e-15)


if __name__ == "__main__":
    unittest.main()
