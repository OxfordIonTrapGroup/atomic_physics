"""
References:
[1] - R. Srinivas, Laser-free trapped-ion quantum logic with a radiofrequency
        magnetic field gradient, PhD Thesis, University of Colorado (2016)
"""

import unittest

import numpy as np

from atomic_physics.ions import mg25
from atomic_physics.utils import field_insensitive_point

S12_31_21_clock_field = 212.8e-4  # [1]
MW_transition_freqs = [
    [3, 2, 1.326456],
    [2, 2, 1.460516],
    [1, 2, 1.573543],
    [2, 1, 1.573432],
    [1, 1, 1.686459],
    [0, 1, 1.786044],
    [0, -1, 1.975445],
]  # [1] Frequency in GHz


class TestMg25Consts(unittest.TestCase):
    def test_s12_clock(self):
        """Look for field-insensitive S1/2 3,1 -> S1/2 2,1 transition at 212.8 G"""
        Mg25 = mg25.Mg25.filter_levels(level_filter=(mg25.S12,))
        ion = Mg25(magnetic_field=S12_31_21_clock_field)

        l_index = ion.get_state_for_F(mg25.S12, F=3, M_F=+1)
        u_index = ion.get_state_for_F(mg25.S12, F=2, M_F=+1)

        # 212.8 G
        model_field_independent_point = field_insensitive_point(
            Mg25, (l_index, u_index), magnetic_field_guess=S12_31_21_clock_field
        )
        self.assertAlmostEqual(
            model_field_independent_point, S12_31_21_clock_field, places=5
        )

    def test_mw_transitions(self):
        """
        Check microwave transition frequencies in the ground state manifold
        """
        Mg25 = mg25.Mg25.filter_levels(level_filter=(mg25.S12,))

        ion = Mg25(magnetic_field=1e-4)
        l_index = ion.get_state_for_F(mg25.S12, F=3, M_F=+1)
        u_index = ion.get_state_for_F(mg25.S12, F=2, M_F=+1)

        model_field_independent_point = field_insensitive_point(
            Mg25, (l_index, u_index), magnetic_field_guess=S12_31_21_clock_field
        )
        ion = Mg25(magnetic_field=model_field_independent_point)

        for transition in MW_transition_freqs:
            freq_model = ion.get_transition_frequency_for_states(
                (
                    ion.get_state_for_F(mg25.S12, F=3, M_F=transition[0]),
                    ion.get_state_for_F(mg25.S12, F=2, M_F=transition[1]),
                ),
            ) / (2 * np.pi * 1e9)

            self.assertAlmostEqual(freq_model, transition[2], places=4)
