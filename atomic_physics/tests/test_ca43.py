import unittest
import numpy as np

from atomic_physics.ions import ca43
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
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12, ca43.D52))
        ion = Ca43(magnetic_field=3e-4)

        l_index = ion.get_state_for_F(ca43.S12, F=4, M_F=+4)
        u_index = ion.get_state_for_F(ca43.D52, F=4, M_F=+3)

        # Start with the field-insensitive transition at around 3G
        B0 = field_insensitive_point(
            Ca43, (l_index, u_index), magnetic_field_guess=3.38e-4
        )
        np.testing.assert_allclose(B0, 3.38e-4, atol=1e-6)

        # Field-insensitive transition at around 5G
        B0 = field_insensitive_point(
            Ca43, (l_index, u_index), magnetic_field_guess=5e-4
        )
        np.testing.assert_allclose(B0, 4.96e-4, atol=1e-6)

    def test_s12_40_31_clock(self):
        """Compare values we compute for the field-insensitive S1/2 4,0 -> S1/2 3,1
        transition at 146.094G to [1].
        """
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12,))
        ion = Ca43(magnetic_field=146.0942e-4)

        s12_40_index = ion.get_state_for_F(ca43.S12, F=4, M_F=0)
        s12_31_index = ion.get_state_for_F(ca43.S12, F=3, M_F=+1)

        B0 = field_insensitive_point(
            Ca43,
            (s12_40_index, s12_31_index),
            magnetic_field_guess=140e-4,
        )
        np.testing.assert_allclose(B0, 146.0942e-4, atol=1e-8)

    def test_s12_41_31_clock(self):
        """Compare values we compute for field-insensitive S1/2 4,1 -> S1/2 3,1
        transition at 288G to [1].
        """
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12,))
        ion = Ca43(magnetic_field=287.7827e-4)

        s12_41_index = ion.get_state_for_F(ca43.S12, F=4, M_F=+1)
        s12_31_index = ion.get_state_for_F(ca43.S12, F=3, M_F=+1)

        B0 = field_insensitive_point(
            Ca43,
            (s12_41_index, s12_31_index),
            magnetic_field_guess=288e-4,
        )
        np.testing.assert_allclose(B0, 287.7827e-4, atol=1e-8)
