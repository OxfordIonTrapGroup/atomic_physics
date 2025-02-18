import unittest

import numpy as np

from atomic_physics.ions.ca43 import Ca43


class TestAtomFactory(unittest.TestCase):
    """Tests for :class:`atomic_physics.core.AtomFactory`."""

    def test_num_states(self):
        ion = Ca43(magnetic_field=146.0942e-4)
        num_states = 0
        for level in ion.level_data.keys():
            num_states += (2 * ion.nuclear_spin + 1) * (2 * level.J + 1)
        assert num_states == Ca43.num_states

    def test_sorting(self):
        """
        Test that the atom factory sorts the levels into energy ordering correctly.
        """
        ion = Ca43(magnetic_field=1.0)
        levels_sorted = sorted(
            ion.level_states.items(), key=lambda item: item[1].frequency
        )

        assert levels_sorted[0][0] == Ca43.S12
        assert levels_sorted[1][0] == Ca43.D32
        assert levels_sorted[2][0] == Ca43.D52
        assert levels_sorted[3][0] == Ca43.P12
        assert levels_sorted[4][0] == Ca43.P32

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
        assert ion.level_states[Ca43.P32].start_index == 0
        assert (
            ion.level_states[Ca43.P12].start_index
            == ion.level_states[Ca43.P32].stop_index
        )
        assert (
            ion.level_states[Ca43.D52].start_index
            == ion.level_states[Ca43.P12].stop_index
        )
        assert (
            ion.level_states[Ca43.D32].start_index
            == ion.level_states[Ca43.D52].stop_index
        )
        assert (
            ion.level_states[Ca43.S12].start_index
            == ion.level_states[Ca43.D32].stop_index
        )

    def test_filtering(self):
        """Test that the level filtering works correctly."""
        levels = (Ca43.S12, Ca43.D32, Ca43.P12, Ca43.P32)
        factory = Ca43.filter_levels(level_filter=levels)
        ion = factory(magnetic_field=1.0)
        assert len(ion.levels) == len(set(ion.levels))
        assert set(ion.levels) == set(levels)

        levels_sorted = sorted(
            ion.level_states.items(), key=lambda item: item[1].frequency
        )

        assert levels_sorted[0][0] == Ca43.S12
        assert levels_sorted[1][0] == Ca43.D32
        assert levels_sorted[2][0] == Ca43.P12
        assert levels_sorted[3][0] == Ca43.P32

        assert levels_sorted[0][1].frequency == 0  # S 1/2 is the ground level
        np.testing.assert_allclose(
            levels_sorted[3][1].frequency - levels_sorted[1][1].frequency,
            ion.transitions["866"].frequency,
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            levels_sorted[1][1].frequency, ion.transitions["733"].frequency, rtol=2e-2
        )
        np.testing.assert_allclose(
            levels_sorted[2][1].frequency, ion.transitions["397"].frequency, rtol=2e-2
        )
        np.testing.assert_allclose(
            levels_sorted[3][1].frequency, ion.transitions["393"].frequency, rtol=2e-2
        )

        # states should be in reverse energy order
        assert ion.level_states[Ca43.P32].start_index == 0
        assert (
            ion.level_states[Ca43.P12].start_index
            == ion.level_states[Ca43.P32].stop_index
        )
        assert (
            ion.level_states[Ca43.D32].start_index
            == ion.level_states[Ca43.P12].stop_index
        )
        assert (
            ion.level_states[Ca43.S12].start_index
            == ion.level_states[Ca43.D32].stop_index
        )

        levels = (Ca43.D32, Ca43.P12, Ca43.P32)

        factory = factory.filter_levels(level_filter=levels)
        ion = factory(magnetic_field=1.0)

        assert len(ion.levels) == len(set(ion.levels))
        assert set(ion.levels) == set(levels)

        levels_sorted = sorted(
            ion.level_states.items(), key=lambda item: item[1].frequency
        )

        assert levels_sorted[0][0] == Ca43.D32
        assert levels_sorted[1][0] == Ca43.P12
        assert levels_sorted[2][0] == Ca43.P32

        assert levels_sorted[0][1].frequency == 0
        np.testing.assert_allclose(
            levels_sorted[1][1].frequency, ion.transitions["866"].frequency, rtol=1e-3
        )
        np.testing.assert_allclose(
            levels_sorted[2][1].frequency, ion.transitions["850"].frequency, rtol=1e-3
        )

        # states should be in reverse energy order
        assert ion.level_states[Ca43.P32].start_index == 0
        assert (
            ion.level_states[Ca43.P12].start_index
            == ion.level_states[Ca43.P32].stop_index
        )
        assert (
            ion.level_states[Ca43.D32].start_index
            == ion.level_states[Ca43.P12].stop_index
        )

        levels = (Ca43.D32,)
        factory = factory.filter_levels(level_filter=levels)
        ion = factory(magnetic_field=1.0)

        assert len(ion.levels) == len(set(ion.levels))
        assert set(ion.levels) == set(levels)

        assert ion.level_states[Ca43.D32].frequency == 0
        assert ion.level_states[Ca43.D32].start_index == 0
