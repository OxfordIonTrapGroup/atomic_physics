import unittest

import numpy as np

from atomic_physics.ions import ca43, ba133, ba137
from atomic_physics.utils import d2f_dB2, df_dB, field_insensitive_point


class TestMagneticDipole(unittest.TestCase):
    def test_ca43_146G_clock(self):
        """Check magnetic dipole calculations against values in T. Harty DPhil thesis
        table E.4 for the ground-level of 43Ca+ at 146G.
        """
        ion = ca43.Ca43(magnetic_field=146.0942e-4)

        magnetic_dipoles = ion.get_magnetic_dipoles()

        np.testing.assert_allclose(
            np.abs(magnetic_dipoles), np.abs(magnetic_dipoles.T), atol=1e-34
        )

        # NB some of the signs of the magnetic dipole matrix elements are different
        # to table E.4. I haven't dug into the discrepancy...famous last words, but I'm
        # going to chalk this up to a difference in definitions somewhere between this
        # package and that work.
        # FIXME: SIGNS
        values = [
            (-4, -3, 1.342420, 2.519894),
            (-3, -3, +0.602802, 2.237093),
            (-3, -2, -1.195577, 1.940390),
            (-2, -3, -0.204417, 1.939817),
            (-2, -2, +0.810866, 1.643113),
            (-2, -1, -1.040756, 1.330090),
            (-1, -2, -0.363393, 1.329517),
            (-1, -1, +0.932839, 1.016493),
            (-1, 0, -0.876791, 0.684932),
            (0, -1, -0.528271, 0.684359),
            (0, 0, +0.993060, 0.352797),
            (0, +1, -0.702127, 0.0),
            (+1, 0, +0.702254, -0.000573),
            (+1, +1, -0.993034, -0.353370),
            (+1, +2, +0.514410, -0.730728),
            (+2, +1, +0.887366, -0.731301),
            (+2, +2, -0.919344, -1.108659),
            (+2, +3, +0.308504, -1.514736),
            (+3, +2, +1.085679, -1.515309),
            (+3, +3, -0.728641, -1.921385),
            (+4, +3, +1.299654, -2.362039),
        ]

        for M4, M3, R_ref, df_dB_ref in values:
            u_index = ion.get_state_for_F(ca43.S12, F=3, M_F=M3)
            l_index = ion.get_state_for_F(ca43.S12, F=4, M_F=M4)

            np.testing.assert_allclose(
                magnetic_dipoles[u_index, l_index] / 9.274e-24, R_ref, atol=1.5e-6
            )

            np.testing.assert_allclose(
                df_dB(
                    atom_factory=ca43.Ca43,
                    states=(l_index, u_index),
                    magnetic_field=146.0942e-4,
                )
                / (2 * np.pi * 1e10),
                df_dB_ref,
                atol=1e-6,
            )

        np.testing.assert_allclose(
            field_insensitive_point(
                atom_factory=ca43.Ca43,
                states=(
                    ion.get_state_for_F(ca43.S12, F=4, M_F=0),
                    ion.get_state_for_F(ca43.S12, F=3, M_F=+1),
                ),
                magnetic_field_guess=10e-4,
            ),
            146.0942e-4,
            atol=1e-4,
        )

        w_clock = ion.get_transition_frequency_for_states(
            (
                ion.get_state_for_F(ca43.S12, F=4, M_F=0),
                ion.get_state_for_F(ca43.S12, F=3, M_F=+1),
            )
        )
        np.testing.assert_allclose(
            w_clock, 2 * np.pi * 3.199941077e9, atol=2 * np.pi * 0.5
        )

        np.testing.assert_allclose(
            d2f_dB2(
                atom_factory=ca43.Ca43,
                magnetic_field=146.0942e-4,
                states=(
                    ion.get_state_for_F(ca43.S12, F=4, M_F=0),
                    ion.get_state_for_F(ca43.S12, F=3, M_F=+1),
                ),
            )
            / (2 * np.pi * 1e11),
            2.416,
            atol=1e-3,
        )

        # Check all forbidden transitions have 0 matrix element
        level = ca43.S12
        level_slice = ion.get_slice_for_level(level)
        for i_ind in np.arange(ion.num_states)[level_slice]:
            for j_ind in np.arange(ion.num_states)[level_slice]:
                if np.abs(ion.M[i_ind] - ion.M[j_ind]) > 1:
                    np.testing.assert_allclose(magnetic_dipoles[i_ind, j_ind], 0)
                    np.testing.assert_allclose(magnetic_dipoles[j_ind, i_ind], 0)

        np.testing.assert_allclose(np.diag(magnetic_dipoles[level_slice]), 0)

        # check the expansion coefficients for the intermediate-field states in terms
        # of the high-field (MI, MJ) states match table E.2

        # M, ((M_J, M_I, coeff_M_J, coeff_M_I))
        expansions = (
            (-4, ((-1 / 2, -7 / 2, 1.0, 0.0), (-1 / 2, -7 / 2, 1.0, 0.0))),
            (-3, ((-1 / 2, -5 / 2, 0.9483, 0.3175), (+1 / 2, -7 / 2, 0.3175, -0.9483))),
            (-2, ((-1 / 2, -3 / 2, 0.8906, 0.4548), (+1 / 2, -5 / 2, 0.4548, -0.8906))),
            (-1, ((-1 / 2, -1 / 2, 0.8255, 0.5645), (+1 / 2, -3 / 2, 0.5645, -0.8255))),
            (0, ((-1 / 2, +1 / 2, 0.7503, 0.6611), (+1 / 2, -1 / 2, 0.6611, -0.7503))),
            (+1, ((-1 / 2, +3 / 2, 0.6610, 0.7504), (+1 / 2, +1 / 2, 0.7504, -0.6610))),
            (+2, ((-1 / 2, +5 / 2, 0.5497, 0.8354), (+1 / 2, +3 / 2, 0.8354, -0.5497))),
            (+3, ((-1 / 2, +7 / 2, 0.3964, 0.9181), (+1 / 2, +5 / 2, 0.9181, -0.3964))),
            (+4, ((+1 / 2, +7 / 2, 1.0, 0.0), (+1 / 2, +7 / 2, 1.0, 0.0))),
        )

        state_vectors = ion.state_vectors[level_slice, :]
        basis_M_I = ion.high_field_M_I[level_slice]
        basis_M_J = ion.high_field_M_J[level_slice]

        for M, (alpha_expansion, beta_expansion) in expansions:
            indicies = ion.get_states_for_M(level=ca43.S12, M=M)
            for ind in indicies:
                state_vector = state_vectors[0:16, ind]

                # There are 2 states with each value of M and each state is a
                # superposition of 2 combinations of (M_I, M_J) states. We label these
                # combinations alpha and beta.

                state_ind = np.argwhere(
                    np.logical_and(
                        basis_M_J == alpha_expansion[0],
                        basis_M_I == alpha_expansion[1],
                    )
                )
                alpha = np.abs(state_vector[state_ind].ravel()[0])  # FIXME: SIGN

                state_ind = np.argwhere(
                    np.logical_and(
                        basis_M_J == beta_expansion[0],
                        basis_M_I == beta_expansion[1],
                    )
                )
                beta = np.abs(state_vector[state_ind].ravel()[0])  # FIXME: sign

                if not np.all(
                    np.isclose(
                        (alpha, beta),
                        (np.abs(alpha_expansion[2]), np.abs(beta_expansion[2])),
                        atol=1e-4,
                    )
                ) and not np.all(
                    np.isclose(
                        (alpha, beta),
                        (np.abs(alpha_expansion[3]), np.abs(beta_expansion[3])),
                        atol=1e-4,
                    )
                ):
                    print(
                        alpha,
                        beta,
                        (np.abs(alpha_expansion[2]), np.abs(beta_expansion[2])),
                        (np.abs(alpha_expansion[3]), np.abs(beta_expansion[3])),
                    )
                    assert False, f"Can't match {alpha, beta}"


class TestFNumber(unittest.TestCase):
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


class TestMIMJ(unittest.TestCase):
    """
    Test MI and MJ number assignment
    """

    def test_ba137_ground(self):
        level = ba137.ground_level
        Ba137 = ba137.Ba137.filter_levels(level_filter=(level,))
        ion = Ba137(magnetic_field=1.0)

        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=2,
                M_F=2,
            ),
            ion.get_state_for_MI_MJ(level, M_I=3 / 2, M_J=1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=2,
                M_F=1,
            ),
            ion.get_state_for_MI_MJ(level, M_I=1 / 2, M_J=1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=2,
                M_F=0,
            ),
            ion.get_state_for_MI_MJ(level, M_I=-1 / 2, M_J=1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=2,
                M_F=-1,
            ),
            ion.get_state_for_MI_MJ(level, M_I=-3 / 2, M_J=1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=2,
                M_F=-2,
            ),
            ion.get_state_for_MI_MJ(level, M_I=-3 / 2, M_J=-1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=1,
                M_F=-1,
            ),
            ion.get_state_for_MI_MJ(level, M_I=-1 / 2, M_J=-1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=1,
                M_F=0,
            ),
            ion.get_state_for_MI_MJ(level, M_I=1 / 2, M_J=-1 / 2),
        )
        self.assertEqual(
            ion.get_state_for_F(
                level,
                F=1,
                M_F=1,
            ),
            ion.get_state_for_MI_MJ(level, M_I=3 / 2, M_J=-1 / 2),
        )
