"""Test F-number assignment"""

import unittest
from atomic_physics.ions import ba133, ba137


class TesLevel(unittest.TestCase):
    """
    Test F-number assignment
    """

    def test_positive_hfs_ground(self):
        level = ba137.ground_level
        Ba137 = ba137.Ba137.filter_levels(level_filter=(level,))
        ion = Ba137(magnetic_field=1e-4)

        # check that states in F=2 have indices 0-4
        for i in range(5):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=2,
                    M_F=-(i - 2),
                ),
                i,
            )

        # check that states in F=1 have indices 5-7
        for i in range(3):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=1,
                    M_F=-(i - 1),
                ),
                7 - i,
            )

    def test_negative_hfs_ground(self):
        level = ba133.ground_level
        level = ba133.ground_level
        Ba133 = ba133.Ba133.filter_levels(level_filter=(level,))
        ion = Ba133(magnetic_field=1e-4)

        # check that F=0, mF=0 has index 0
        self.assertEqual(ion.get_state_for_F(level, F=0, M_F=0), 0)

        # check that states in F=1 have indices 1-3
        for i in range(3):
            self.assertEqual(ion.get_state_for_F(level, F=1, M_F=i - 1), 3 - i)

    def test_hfs_metastable(self):
        level = ba137.shelf
        Ba137 = ba137.Ba137.filter_levels(level_filter=(level,))
        ion = Ba137(magnetic_field=1e-9)

        # check that states in F=1 have indices 0-2
        for i in range(3):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=1,
                    M_F=-(i - 1),
                ),
                i,
            )

        # check that states in F=2 have indices 3-7
        for i in range(5):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=2,
                    M_F=-(i - 2),
                ),
                i + 3,
            )

        # check that states in F=3 have indices 8-14
        for i in range(7):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=3,
                    M_F=-(i - 3),
                ),
                i + 8,
            )

        # check that states in F=4 have indices 15-23
        for i in range(9):
            self.assertEqual(
                ion.get_state_for_F(
                    level,
                    F=4,
                    M_F=-(i - 4),
                ),
                i + 15,
            )
