import unittest

import numpy as np

from atomic_physics.ions.ca43 import Ca43

from .utils import wigner_3j, wigner_6j


class TestGamma(unittest.TestCase):
    def test_LF(self):
        """Check that, in the low-field, our scattering rates match a more direct
        calculation.
        """
        ion = Ca43(magnetic_field=1e-8)
        Gamma_ion = np.abs(ion.get_electric_multipoles()) ** 2
        Idim = int(np.rint(2 * ion.nuclear_spin + 1))
        Gamma = np.zeros((ion.num_states, ion.num_states))

        for name, transition in ion.transitions.items():
            A = transition.einstein_A
            lower = transition.lower
            upper = transition.upper
            Ju = upper.J
            Jl = lower.J
            Jdim_l = int(np.rint(2 * Jl + 1))
            l_dim = Idim * Jdim_l

            dJ = Ju - Jl
            dL = upper.L - lower.L
            if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                order = 1
            elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                order = 2
            else:
                raise ValueError("Unsupported transition order {}".format(order))

            subspace = np.r_[
                ion.get_slice_for_level(lower), ion.get_slice_for_level(upper)
            ]

            for l_ind in list(subspace[:l_dim]):
                for u_ind in list(subspace[l_dim:]):
                    Fl = ion.F[l_ind]
                    Fu = ion.F[u_ind]
                    Ml = ion.M[l_ind]
                    Mu = ion.M[u_ind]
                    q = Mu - Ml
                    if q not in range(-order, order + 1):
                        continue

                    Gamma[l_ind, u_ind] = A * (
                        (2 * Ju + 1)
                        * (2 * Fl + 1)
                        * (2 * Fu + 1)
                        * (wigner_3j(Fu, order, Fl, -Mu, q, Ml)) ** 2
                        * (wigner_6j(Ju, ion.nuclear_spin, Fu, Fl, order, Jl) ** 2)
                    )

            subspace = np.ix_(subspace, subspace)
            scale = np.max(np.max(np.abs(Gamma[subspace])))
            eps = np.max(np.max(np.abs(Gamma[subspace] - Gamma_ion[subspace])))
            self.assertLess(eps / scale, 1e-4)

    def test_HF(self):
        """Check that, in the high-field, our scattering rates match a more
        direct calculation."""
        ion = Ca43(magnetic_field=1000)
        Gamma_ion = np.abs(ion.get_electric_multipoles()) ** 2
        Idim = int(np.rint(2 * ion.nuclear_spin + 1))
        Gamma = np.zeros((ion.num_states, ion.num_states))

        for _, transition in ion.transitions.items():
            A = transition.einstein_A
            lower = transition.lower
            upper = transition.upper
            Ju = upper.J
            Jl = lower.J
            Jdim_l = int(np.rint(2 * Jl + 1))
            l_dim = Idim * Jdim_l

            dJ = Ju - Jl
            dL = upper.L - lower.L
            if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                order = 1
            elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                order = 2
            else:
                raise ValueError("Unsupported transition order {}".format(order))

            subspace = np.r_[
                ion.get_slice_for_level(lower), ion.get_slice_for_level(upper)
            ]

            for l_ind in list(subspace[:l_dim]):
                for u_ind in list(subspace[l_dim:]):
                    if ion.M_I[l_ind] != ion.M_I[u_ind]:
                        continue
                    M_l = ion.M_J[l_ind]
                    M_u = ion.M_J[u_ind]
                    q = M_u - M_l
                    if q not in range(-order, order + 1):
                        continue
                    Gamma[l_ind, u_ind] = (
                        A * (2 * Ju + 1) * (wigner_3j(Ju, order, Jl, -M_u, q, M_l)) ** 2
                    )

            subspace = np.ix_(subspace, subspace)
            scale = np.max(np.max(np.abs(Gamma[subspace])))
            eps = np.max(np.max(np.abs(Gamma[subspace] - Gamma_ion[subspace])))
            self.assertTrue(eps / scale < 1e-4)


if __name__ == "__main__":
    unittest.main()
