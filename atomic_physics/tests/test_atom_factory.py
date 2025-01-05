import unittest

import numpy as np

from atomic_physics.ions import ca43


class TestAtomFactory(unittest.TestCase):
    """Tests for :class:`atomic_physics.core.AtomFactory`."""

    def test_sorting(self):
        """
        Test that the atom factory sorts the levels into energy ordering correctly.
        """
        ion = ca43.Ca43(magnetic_field=1.0)
        levels_sorted = sorted(
            ion.level_states.items(), key=lambda item: item[1].frequency
        )

        assert levels_sorted[0][0] == ca43.S12
        assert levels_sorted[1][0] == ca43.D32
        assert levels_sorted[2][0] == ca43.D52
        assert levels_sorted[3][0] == ca43.P12
        assert levels_sorted[4][0] == ca43.P32

        # check all levels have the right energy
        assert levels_sorted[0][1].frequency == 0  # S 1/2 is the ground level
        np.testing.assert_allclose(
            levels_sorted[3][1].frequency - levels_sorted[1][1].frequency,
            ion.transitions["866"].frequency,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            levels_sorted[1][1].frequency, ion.transitions["733"].frequency, rtol=1e-3
        )
        np.testing.assert_allclose(
            levels_sorted[2][1].frequency, ion.transitions["729"].frequency, rtol=1e-3
        )
        np.testing.assert_allclose(
            levels_sorted[3][1].frequency, ion.transitions["397"].frequency, rtol=1e-3
        )
        np.testing.assert_allclose(
            levels_sorted[4][1].frequency, ion.transitions["393"].frequency, rtol=1e-3
        )

        I_dim = 2 * ion.nuclear_spin + 1
        for level, states in ion.level_states.items():
            J_dim = 2 * level.J + 1
            assert states.num_states == np.rint(I_dim * J_dim)
            assert states.stop_index == states.start_index + states.num_states

        # states should be in reverse energy order
        assert ion.level_states[ca43.P32].start_index == 0
        assert (
            ion.level_states[ca43.P12].start_index
            == ion.level_states[ca43.P32].stop_index
        )
        assert (
            ion.level_states[ca43.D52].start_index
            == ion.level_states[ca43.P12].stop_index
        )
        assert (
            ion.level_states[ca43.D32].start_index
            == ion.level_states[ca43.D52].stop_index
        )
        assert (
            ion.level_states[ca43.S12].start_index
            == ion.level_states[ca43.D32].stop_index
        )

    def test_filtering(self):
        """Test that the level filtering works correctly."""
        pass
