import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics import operators
from atomic_physics.core import AtomFactory, Level, LevelData
from atomic_physics.ions import ba133, ba137, ca43


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

    def test_F_positive_ground(self):
        """Check we're assigning the right value of F to each state.

        This test looks at the ground level of 137Ba+, which has a positive hyperfine A
        coefficient.
        """
        level = ba137.ground_level
        Ba137 = ba137.Ba137.filter_levels(level_filter=(level,))

        # Make sure we get the relationships right over a range of fields
        for magnetic_field in [1e-4, 100e-4, 1000e-4, 1, 10]:
            ion = Ba137(magnetic_field=magnetic_field)

            # 137Ba+ has I=3/2 so the ground level has F=1 and F=2. A is positive so
            # F=2 has higher energy.

            # check that states in F=2 have indices 0-4
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+2, M_F=M_F) for M_F in range(-2, +3)]
            )
            assert all(inds == np.arange(5)[::-1])

            # check that states in F=1 have indices 5-7
            # states with higher M_F have lower energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+1, M_F=M_F) for M_F in range(-1, +2)]
            )
            assert all(inds == np.arange(3) + 5)

    def test_F_negative_ground(self):
        """Check we're assigning the right value of F to each state.

        This test looks at the ground level of 133Ba+, which has a negative hyperfine A
        coefficient.
        """
        level = ba133.ground_level
        Ba133 = ba133.Ba133.filter_levels(level_filter=(level,))

        # Make sure we get the relationships right over a range of fields
        for magnetic_field in [1e-4, 100e-4, 1000e-4, 1, 10]:
            ion = Ba133(magnetic_field=magnetic_field)

            # 137Ba+ has I=1/2 so the ground level has F=0 and F=1. A is negative so
            # F=1 has higher energy.

            # check that F=0, mF=0 has index 0
            self.assertEqual(ion.get_state_for_F(level, F=0, M_F=0), 0)

            # check that states in F=1 have indices 1-3
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+1, M_F=M_F) for M_F in range(-1, +2)]
            )
            assert all(inds == np.arange(3)[::-1] + 1)

    def test_F_D_level(self):
        """Check we're assigning the right value of F to each state.

        This test looks at the D level of 137Ba+, which has a positive hyperfine A
        coefficient and a negative B coefficient.
        """
        level = ba137.shelf
        Ba137 = ba137.Ba137.filter_levels(level_filter=(level,))

        for magnetic_field in [1e-6, 1e-4, 10e-4, 100]:
            ion = Ba137(magnetic_field=1e-9)

            # 137Ba+ has I=3/2 so the ground level F=1, F=2, F=3, F=4. A is negative and
            # (just) outweighs B (which is positive) so F=1 has highest energy.

            # check that states in F=1 have indices 0-2
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+1, M_F=M_F) for M_F in range(-1, +2)]
            )
            assert all(inds == np.arange(3)[::-1])

            # check that states in F=2 have indices 3-7
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+2, M_F=M_F) for M_F in range(-2, +3)]
            )
            assert all(inds == np.arange(5)[::-1] + 3)

            # check that states in F=3 have indices 8-14
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+3, M_F=M_F) for M_F in range(-3, +4)]
            )
            assert all(inds == np.arange(7)[::-1] + 8)

            # check that states in F=4 have indices 15-23
            # states with higher M_F have greater energy
            inds = np.array(
                [ion.get_state_for_F(level, F=+4, M_F=M_F) for M_F in range(-4, +5)]
            )
            assert all(inds == np.arange(9)[::-1] + 15)

    def test_F_large_quadrupole(self):
        """Check we're assigning the right value of F to each state.

        This test looks at the case where the nuclear quadrupole term dominates the
        dipole term.
        """
        level = Level(n=1, S=+1 / 2, L=2, J=+5 / 2)
        nuclear_spin = +3 / 2

        for Ahfs, Bhfs in (
            (-10e6, +100e6),
            (+10e6, -100e6),
            (+10e6, 100e6),
            (-10e6, -100e6),
        ):
            level_data = LevelData(
                level=level, Ahfs=Ahfs * consts.h, Bhfs=Bhfs * consts.h, g_I=(2 / 3)
            )
            factory = AtomFactory(
                level_data=(level_data,), transitions=tuple(), nuclear_spin=nuclear_spin
            )
            # level = ca43.ground_level
            # factory = ca43.Ca43.filter_levels(level_filter=(level, ))
            atom = factory(magnetic_field=1e-9)

            # nuclear_spin = atom.nuclear_spin

            # In the low field we should have <F^2> = F(F+1)
            I_dim = int(np.rint(2 * nuclear_spin + 1))
            J_dim = int(np.rint(2 * level.J + 1))

            Jp = np.kron(operators.Jp(level.J), np.identity(I_dim))
            Jm = np.kron(operators.Jm(level.J), np.identity(I_dim))
            Jz = np.kron(operators.Jz(level.J), np.identity(I_dim))

            Ip = np.kron(np.identity(J_dim), operators.Jp(atom.nuclear_spin))
            Im = np.kron(np.identity(J_dim), operators.Jm(atom.nuclear_spin))
            Iz = np.kron(np.identity(J_dim), operators.Jz(atom.nuclear_spin))

            Fz = Iz + Jz
            Fp = Ip + Jp
            Fm = Im + Jm

            F_2_op = Fz @ Fz + (1 / 2) * (Fp @ Fm + Fm @ Fp)
            F_2 = np.diag(
                atom.state_vectors.conj().T @ F_2_op @ atom.state_vectors
            )  # <F^2>

            F = 0.5 * (np.sqrt(1 + 4 * F_2) - 1)  # <F^2> = f * (f + 1)
            np.testing.assert_allclose(atom.F, F, atol=0.1)
