"""
References:
[1] - J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
[2] - R. K. Hanley, et al. PHYSICAL REVIEW A 104, 052804 (2021)
[3] - M. Weber, High-Fidelity, Near-Field Microwave Gates in a Cryogenic Surface Trap,
      DPhil Thesis (2022)
"""

import unittest

from atomic_physics.ions import ca43
from atomic_physics.utils import field_insensitive_point

S12_44_D52_43_clock_fields = (3.38e-4, 4.96e-4)  # [1]
S12_40_31_clock_field = 146.094e-4  # [2]
S12_41_31_clock_field = 288e-4  # [3]


class TestCa43Consts(unittest.TestCase):
    def test_s12_d52_clock(self):
        """Look for field-insensitive S1/2 4,4 -> D5/2 4,3 transition at 3.38 G and
        4.96 G
        """
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12, ca43.D52))
        ion = Ca43(magnetic_field=S12_44_D52_43_clock_fields[0])

        l_index = ion.get_state_for_F(ca43.S12, F=4, M_F=+4)
        u_index = ion.get_state_for_F(ca43.D52, F=4, M_F=+3)

        # 3.38 G
        model_field_independent_point_1 = field_insensitive_point(
            Ca43, (l_index, u_index), magnetic_field_guess=S12_44_D52_43_clock_fields[0]
        )
        self.assertAlmostEqual(
            model_field_independent_point_1, S12_44_D52_43_clock_fields[0], places=6
        )

        # 4.96 G
        model_field_independent_point_2 = field_insensitive_point(
            Ca43, (l_index, u_index), magnetic_field_guess=S12_44_D52_43_clock_fields[1]
        )
        # FIXME: field_insensitive_point() gives 4.947… G, which does not match the
        # value from [1] to the number of significant digits given there. This is not a
        # straightforward root finding or finite-difference accuracy issue.
        self.assertAlmostEqual(
            model_field_independent_point_2, S12_44_D52_43_clock_fields[1], places=5
        )

    def test_s12_40_31_clock(self):
        """Look for field-insensitive S1/2 4,0 -> S1/2 3,1 transition at 146.094G"""
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12,))
        ion = Ca43(magnetic_field=S12_40_31_clock_field)

        s12_40_index = ion.get_state_for_F(ca43.S12, F=4, M_F=0)
        s12_31_index = ion.get_state_for_F(ca43.S12, F=3, M_F=+1)

        model_field_independent_point_40_31 = field_insensitive_point(
            Ca43,
            (s12_40_index, s12_31_index),
            magnetic_field_guess=S12_40_31_clock_field,
        )
        self.assertAlmostEqual(
            model_field_independent_point_40_31, S12_40_31_clock_field, places=7
        )

    def test_s12_41_31_clock(self):
        """Look for field-insensitive S1/2 4,1 -> S1/2 3,1 transition at 288G"""
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.S12,))
        ion = Ca43(magnetic_field=S12_41_31_clock_field)

        s12_41_index = ion.get_state_for_F(ca43.S12, F=4, M_F=+1)
        s12_31_index = ion.get_state_for_F(ca43.S12, F=3, M_F=+1)

        model_field_independent_point_41_31 = field_insensitive_point(
            Ca43,
            (s12_41_index, s12_31_index),
            magnetic_field_guess=S12_41_31_clock_field,
        )
        self.assertAlmostEqual(
            model_field_independent_point_41_31, S12_41_31_clock_field, places=4
        )
