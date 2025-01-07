import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics.core import Level, LevelData


class TestLevelData(unittest.TestCase):
    """Tests for :class:`atomic_physics.core.LevelData`."""

    def test_g_J(self):
        """
        Test that ``LevelData`` calculates g factors correctly.
        """
        # L, S, J
        levels = (
            (0, 1 / 2, 1 / 2),
            (1, 1 / 2, 1 / 2),
            (1, 1 / 2, 3 / 2),
            (2, 1 / 2, 5 / 2),
            (2, 7 / 2, 11 / 2),
            (2, 7 / 2, 3 / 2),
        )

        def Lande(L, S, J):
            g_L = 1
            g_S = -consts.physical_constants["electron g factor"][0]

            return g_L * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (
                2 * J * (J + 1)
            ) + g_S * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

        for L, S, J in levels:
            level = Level(n=0, L=L, S=S, J=J)
            data = LevelData(level=level, g_I=0, Ahfs=0, Bhfs=0)

            np.testing.assert_allclose(data.g_J, Lande(L=L, S=S, J=J), rtol=5e-3)
