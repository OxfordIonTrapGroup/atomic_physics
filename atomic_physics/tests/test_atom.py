import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics.ions import ca43
from atomic_physics.utils import d2f_dB2, df_dB, field_insensitive_point


class TestCa43(unittest.TestCase):
    def test_ca43_146G_clock(self):
        """Check calculations against values in [1] appendix E for the ground-level of
        43Ca+ at 146G.

        This checks: magnetic dipole matrix elements, state expansion coefficients,
        transition frequencies and sensitivities

        NB if |psi> is an angular momentum eigenstate with eigenvalue M, so is
        e^(i phi)|psi>. In other words, the expansion coefficients - and hence all
        matrix elements - are only determined up to an overall phase factor. In appendix
        E.2 a sign convention was applied when tabulating the expansion coefficients
        however this sign convention was not followed during the calculation of the
        matrix elements tabulated in table E.4 - essentially, the sign convention was
        just applied as a post-processing step when producing E.2, but not anywhere
        else. As a result, all we can do is compare absolute values of matrix elements,
        not signs.

        [1] Thomas Harty DPhil Thesis (2013).
        """
        ion = ca43.Ca43(magnetic_field=146.0942e-4)
        level_slice = ion.get_slice_for_level(ca43.ground_level)
        uB = consts.physical_constants["Bohr magneton"][0]

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
                alpha = state_vector[state_ind].ravel()[0]

                state_ind = np.argwhere(
                    np.logical_and(
                        basis_M_J == beta_expansion[0],
                        basis_M_I == beta_expansion[1],
                    )
                )
                beta = state_vector[state_ind].ravel()[0]

                # Apply the sign convention adopted in [1]
                if alpha < 0:
                    alpha = -alpha
                    beta = -beta

                closest = np.argmin(np.abs(alpha - [alpha_expansion[2:]]))
                alpha_ref = alpha_expansion[closest + 2]
                beta_ref = beta_expansion[closest + 2]
                np.testing.assert_allclose(
                    (alpha, beta),
                    (alpha_ref, beta_ref),
                    atol=1e-4,
                )

        # Compare magnetic dipole matrix element, transition frequency and sensitivity
        # against the values in [1] table E.4
        magnetic_dipoles = ion.get_magnetic_dipoles()

        values = [
            (-4, -3, 1.342420, 2.519894),
            (-3, -3, 0.602802, 2.237093),
            (-3, -2, 1.195577, 1.940390),
            (-2, -3, 0.204417, 1.939817),
            (-2, -2, 0.810866, 1.643113),
            (-2, -1, 1.040756, 1.330090),
            (-1, -2, 0.363393, 1.329517),
            (-1, -1, 0.932839, 1.016493),
            (-1, 0, 0.876791, 0.684932),
            (0, -1, 0.528271, 0.684359),
            (0, 0, 0.993060, 0.352797),
            (0, +1, 0.702127, 0.0),
            (+1, 0, 0.702254, -0.000573),
            (+1, +1, 0.993034, -0.353370),
            (+1, +2, 0.514410, -0.730728),
            (+2, +1, 0.887366, -0.731301),
            (+2, +2, 0.919344, -1.108659),
            (+2, +3, 0.308504, -1.514736),
            (+3, +2, 1.085679, -1.515309),
            (+3, +3, 0.728641, -1.921385),
            (+4, +3, 1.299654, -2.362039),
        ]

        for M4, M3, R_ref, df_dB_ref in values:
            u_index = ion.get_state_for_F(ca43.S12, F=3, M_F=M3)
            l_index = ion.get_state_for_F(ca43.S12, F=4, M_F=M4)

            np.testing.assert_allclose(
                np.abs(magnetic_dipoles[u_index, l_index]) / uB, R_ref, atol=1e-6
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
        # Check that Rnm = (-1)^q Rmn
        level = ca43.S12
        level_slice = ion.get_slice_for_level(level)
        for i_ind in np.arange(ion.num_states)[level_slice]:
            for j_ind in np.arange(ion.num_states)[level_slice]:
                upper_ind = min(i_ind, j_ind)
                lower_ind = max(i_ind, j_ind)
                q = int(np.rint(ion.M[upper_ind] - ion.M[lower_ind]))
                if np.abs(q) > 1:
                    np.testing.assert_allclose(magnetic_dipoles[i_ind, j_ind], 0)
                    np.testing.assert_allclose(magnetic_dipoles[j_ind, i_ind], 0)

                np.testing.assert_allclose(
                    magnetic_dipoles[upper_ind, lower_ind],
                    (-1) ** (q) * magnetic_dipoles[lower_ind, upper_ind],
                )

        np.testing.assert_allclose(np.diag(magnetic_dipoles[level_slice]), 0)
