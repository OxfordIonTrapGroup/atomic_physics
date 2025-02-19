"""Test Spin 1/2 nuclei"""

import unittest

import numpy as np
import scipy.constants as consts

from atomic_physics.core import Level
from atomic_physics.ions.ba133 import Ba133


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


def _gF(F, J, nuclear_spin, gJ):
    return (
        gJ
        * (F * (F + 1) + J * (J + 1) - nuclear_spin * (nuclear_spin + 1))
        / (2 * F * (F + 1))
    )


class TestSpinHalf(unittest.TestCase):
    """Test for spin-half nuclei"""

    def test_hyperfine(self):
        level = Ba133.ground_level
        factory = Ba133.filter_levels(level_filter=(level,))
        ion = factory(0.1e-4)

        E_0_0 = ion.state_energies[ion.get_state_for_F(level, F=0, M_F=0)]
        E_1_0 = ion.state_energies[ion.get_state_for_F(level, F=1, M_F=0)]

        Ahfs = 2 * np.pi * 9925.45355459e6

        self.assertAlmostEqual(abs(E_0_0 - E_1_0) / 1e6, Ahfs / 1e6, 4)

    def test_zeeman(self):
        level = Ba133.ground_level
        factory = Ba133.filter_levels(level_filter=(level,))
        B = 10e-4
        ion = factory(B)

        E_1_0 = ion.state_energies[ion.get_state_for_F(level, F=1, M_F=0)]
        E_1_1 = ion.state_energies[ion.get_state_for_F(level, F=1, M_F=1)]

        gJ = Lande_g(level)
        gF = _gF(1, 1 / 2, 1 / 2, gJ)
        muB = consts.physical_constants["Bohr magneton"][0]
        E_Zeeman = gF * muB * B / consts.hbar

        self.assertAlmostEqual(abs(E_1_0 - E_1_1) / 1e6, E_Zeeman / 1e6, 0)
