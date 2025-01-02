import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics.ions import ca43


class TestAtom(unittest.TestCase):
    """Tests for :class:`atomic_physics.core.Atom`.

    References:
        [1] Thomas Harty DPhil Thesis (2013).
    """

    def test_state_vectors(self):
        """
        Check the calculation for `state_vectors` against values for the ground-level of
        43Ca+ at 146G using [1] table E.2.
        """
        ion = ca43.Ca43(magnetic_field=146.0942e-4)
        level_slice = ion.get_slice_for_level(ca43.ground_level)

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

        matched = np.full((len(expansions), 2), False, dtype=bool)
        for expansion_idx, (M, (alpha_expansion, beta_expansion)) in enumerate(
            expansions
        ):
            indicies = ion.get_states_for_M(level=ca43.S12, M=M)

            for ind in indicies:
                state_vector = state_vectors[:, ind]

                # Each |M> state is a superposition of two high-field states:
                # alpha*|M_J=-1/2, M_I=M+1/2> + beta*|M_J=+1/2, M_I=M-1/2>
                # NB if |psi> is an eigenstate then so is -|psi> so the overall sign of
                # alpha and beta is not uniquely-defined (although the sign of their ratio
                # is). [1] adopts a convention where alpha is always positive; we do the
                # same here for the sake of comparison.

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
                matched[expansion_idx, closest] = True

        # Paranoia check: the above test could theoretically pass with us incorrectly
        # finding the same expansion coefficients for two states. Check this isn't the
        # case

        # There is only 1 way of making the stretched state!
        if any(matched[0, :]):
            matched[0, :] = True
        if any(matched[-1, :]):
            matched[-1, :] = True

        assert np.all(matched)

    def test_magnetic_dipoles(self):
        """
        Check the calculation for `state_vectors` for 43Ca+ at 146G against [1] table E.4.
        """
        uB = consts.physical_constants["Bohr magneton"][0]
        ion = ca43.Ca43(magnetic_field=146.0942e-4)
        level_slice = ion.get_slice_for_level(ca43.ground_level)

        magnetic_dipoles = ion.get_magnetic_dipoles()

        values = (
            (-4, -3, 1.342420),
            (-3, -3, 0.602802),
            (-3, -2, 1.195577),
            (-2, -3, 0.204417),
            (-2, -2, 0.810866),
            (-2, -1, 1.040756),
            (-1, -2, 0.363393),
            (-1, -1, 0.932839),
            (-1, 0, 0.876791),
            (0, -1, 0.528271),
            (0, 0, 0.993060),
            (0, +1, 0.702127),
            (+1, 0, 0.702254),
            (+1, +1, 0.993034),
            (+1, +2, 0.514410),
            (+2, +1, 0.887366),
            (+2, +2, 0.919344),
            (+2, +3, 0.308504),
            (+3, +2, 1.085679),
            (+3, +3, 0.728641),
            (+4, +3, 1.299654),
        )

        for M4, M3, R_ref in values:
            u_index = ion.get_state_for_F(ca43.S12, F=3, M_F=M3)
            l_index = ion.get_state_for_F(ca43.S12, F=4, M_F=M4)

            np.testing.assert_allclose(
                np.abs(magnetic_dipoles[u_index, l_index]) / uB, R_ref, atol=1e-6
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
