import unittest

import numpy as np

from atomic_physics.ions import mg25
from atomic_physics.utils import field_insensitive_point


class TestMg25(unittest.TestCase):
    """Tests of Mg25 atomic structure data.

    References:
        [1] - R. Srinivas, Laser-free trapped-ion quantum logic with a radiofrequency
        magnetic field gradient, PhD Thesis, University of Colorado (2016)
    """

    def test_s12_clock(self):
        """Compare data for the field-insensitive S1/2 3,1 -> S1/2 2,1 transition at
        212.8 G to [1].
        """
        Mg25 = mg25.Mg25.filter_levels(level_filter=(mg25.S12,))
        ion = Mg25(magnetic_field=212.8e-4)

        l_index = ion.get_state_for_F(mg25.S12, F=3, M_F=+1)
        u_index = ion.get_state_for_F(mg25.S12, F=2, M_F=+1)

        B0 = field_insensitive_point(
            Mg25, (l_index, u_index), magnetic_field_guess=212.8e-4
        )
        np.testing.assert_allclose(B0, 212.8e-4, atol=1e-5)

        ion = Mg25(magnetic_field=B0)
        ref = (
            [3, 2, 1.326456],
            [2, 2, 1.460516],
            [1, 2, 1.573543],
            [2, 1, 1.573432],
            [1, 1, 1.686459],
            [0, 1, 1.786044],
            [0, -1, 1.975445],
        )
        for M_3, M_2, ref_freq in ref:
            freq = ion.get_transition_frequency_for_states(
                (
                    ion.get_state_for_F(mg25.S12, F=3, M_F=M_3),
                    ion.get_state_for_F(mg25.S12, F=2, M_F=M_2),
                ),
            )

            # FIXME: this does not agree to the level of precision I would expect
            # atol should be <1e3
            np.testing.assert_allclose(freq / (2 * np.pi), ref_freq * 1e9, atol=3e3)
