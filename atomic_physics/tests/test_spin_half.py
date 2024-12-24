"""Test Spin 1/2 nuclei"""

import unittest

import numpy as np
import scipy.constants as consts

from atomic_physics.common import Level
from atomic_physics.ions import ba133


def Lande_g(level: Level):
    """Returns the Lande g factor for a level."""
    gL = 1
    gS = -consts.physical_constants["electron g factor"][0]

    S = level.S
    J = level.J
    L = level.L

    gJ = gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) + gS * (
        J * (J + 1) + S * (S + 1) - L * (L + 1)
    ) / (2 * J * (J + 1))
    return gJ


def _gF(F, J, I, gJ):
    return gJ * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))


class TestSpinHalf(unittest.TestCase):
    """Test setB() for spin-half nuclei"""

    def test_hyperfine(self):
        level = ba133.ground_level
        ion = ba133.Ba133(level_filter=[level])
        B = 0.1e-4
        ion.setB(B)
        E_0_0 = ion.E[ion.get_index(level, 0, F=0)]
        E_1_0 = ion.E[ion.get_index(level, 0, F=1)]

        Ahfs = 2 * np.pi * 9925.45355459e6

        self.assertAlmostEqual(abs(E_0_0 - E_1_0) / 1e6, Ahfs / 1e6, 4)

    def test_zeeman(self):
        level = ba133.ground_level
        ion = ba133.Ba133(level_filter=[level])
        B = 10e-4
        ion.setB(B)
        E_1_0 = ion.E[ion.get_index(level, 0, F=1)]
        E_1_1 = ion.E[ion.get_index(level, 1, F=1)]

        gJ = Lande_g(level)
        gF = _gF(1, 1 / 2, 1 / 2, gJ)
        muB = consts.physical_constants["Bohr magneton"][0]
        E_Zeeman = gF * muB * B / consts.hbar

        self.assertAlmostEqual(abs(E_1_0 - E_1_1) / 1e6, E_Zeeman / 1e6, 0)
