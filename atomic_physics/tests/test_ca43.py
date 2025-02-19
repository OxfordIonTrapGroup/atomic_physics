import unittest

import numpy as np

from atomic_physics.ions.ca43 import Ca43
from atomic_physics.utils import field_insensitive_point


class TestCa43(unittest.TestCase):
    """Tests for calcium 43 atomic structure.

    References:
        [1] J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
        [2] T. Harty DPhil Thesis (2013)
    """

    def test_s12_d52_clock(self):
        """Compare values we compute for the for field-insensitive
        S1/2 4,4 -> D5/2 4,3 transition at 3.38 G and 4.96 G with [1].
        """
        factory = Ca43.filter_levels(level_filter=(Ca43.S12, Ca43.D52))
        # Start with the field-insensitive transition at around 3G
        B0 = field_insensitive_point(
            factory,
            level_0=Ca43.S12,
            F_0=4,
            M_F_0=+4,
            level_1=Ca43.D52,
            F_1=4,
            M_F_1=+3,
            magnetic_field_guess=3.38e-4,
        )
        np.testing.assert_allclose(B0, 3.38e-4, atol=1e-6)

        # Field-insensitive transition at around 5G
        B0 = field_insensitive_point(
            factory,
            level_0=Ca43.S12,
            F_0=4,
            M_F_0=+4,
            level_1=Ca43.D52,
            F_1=4,
            M_F_1=+3,
            magnetic_field_guess=5e-4,
        )
        np.testing.assert_allclose(B0, 4.96e-4, atol=1e-6)

    def test_s12_40_31_clock(self):
        """Compare values we compute for the field-insensitive S1/2 4,0 -> S1/2 3,1
        transition at 146.094G to [1].
        """
        factory = Ca43.filter_levels(level_filter=(Ca43.S12,))
        B0 = field_insensitive_point(
            factory,
            level_0=Ca43.S12,
            F_0=4,
            M_F_0=0,
            level_1=Ca43.S12,
            F_1=3,
            M_F_1=+1,
            magnetic_field_guess=140e-4,
        )
        np.testing.assert_allclose(B0, 146.0942e-4, atol=1e-8)

    def test_s12_41_31_clock(self):
        """Compare values we compute for field-insensitive S1/2 4,1 -> S1/2 3,1
        transition at 288G to [1].
        """
        factory = Ca43.filter_levels(level_filter=(Ca43.S12,))
        B0 = field_insensitive_point(
            factory,
            level_0=Ca43.S12,
            F_0=4,
            M_F_0=+1,
            level_1=Ca43.S12,
            F_1=3,
            M_F_1=+1,
            magnetic_field_guess=288e-4,
        )
        np.testing.assert_allclose(B0, 287.7827e-4, atol=1e-8)
