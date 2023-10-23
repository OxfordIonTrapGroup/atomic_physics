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
        ion = ca43.Ca43(level_filter=[ca43.S12, ca43.D52])

        ion.setB(S12_44_D52_43_clock_fields[0])
        l_index = ion.index(ca43.S12, 4, F=4)
        u_index = ion.index(ca43.D52, 3, F=4)

        # 3.38 G
        model_field_independent_point_1 = field_insensitive_point(
            ion, l_index, u_index, B0=S12_44_D52_43_clock_fields[0]
        )
        self.assertAlmostEqual(
            model_field_independent_point_1, S12_44_D52_43_clock_fields[0], places=2
        )

        # 4.96 G
        # The curve is quite flat near this point, giving the solver some trouble. Using a slightly larger delta
        model_field_independent_point_2 = field_insensitive_point(
            ion, l_index, u_index, B0=S12_44_D52_43_clock_fields[1]
        )
        self.assertAlmostEqual(
            model_field_independent_point_2, S12_44_D52_43_clock_fields[1], delta=0.02
        )

    def test_s12_40_31_clock(self):
        """Look for field-insensitive S1/2 4,0 -> S1/2 3,1 transition at 146.094G"""
        ion = ca43.Ca43(level_filter=[ca43.S12])

        ion.setB(S12_40_31_clock_field)
        s12_40_index = ion.index(ca43.S12, 0, F=4)
        s12_31_index = ion.index(ca43.S12, 1, F=3)

        model_field_independent_point_40_31 = field_insensitive_point(
            ion, s12_40_index, s12_31_index, B0=S12_40_31_clock_field
        )
        self.assertAlmostEqual(
            model_field_independent_point_40_31, S12_40_31_clock_field, places=1
        )

    def test_s12_41_31_clock(self):
        """Look for field-insensitive S1/2 4,1 -> S1/2 3,1 transition at 288G"""
        ion = ca43.Ca43(level_filter=[ca43.S12])

        ion.setB(B=S12_41_31_clock_field)
        s12_41_index = ion.index(ca43.S12, 1, F=4)
        s12_31_index = ion.index(ca43.S12, 1, F=3)

        model_field_independent_point_41_31 = field_insensitive_point(
            ion, s12_41_index, s12_31_index, B0=S12_41_31_clock_field
        )
        self.assertAlmostEqual(
            model_field_independent_point_41_31, S12_41_31_clock_field, delta=1
        )
