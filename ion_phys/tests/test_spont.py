import numpy as np
import unittest
from sympy.physics.wigner import wigner_3j, wigner_6j
from ion_phys.ions.ca43 import Ca43
from ion_phys.rate_equations import Rates


class TestSpont(unittest.TestCase):
    def test_LF(self):
        """ Check that, in the low-field, our spontaneous scattering rates
        match a more direct calculation. """
        ion = Ca43(1e-8)
        rates = Rates(ion)
        I = ion.I
        Idim = int(np.rint(2*I+1))
        spont_rates = rates.get_spont()
        spont = np.zeros((ion.num_states, ion.num_states))

        for name, transition in ion.transitions.items():
            A = transition.A
            lower = transition.lower
            upper = transition.upper
            Ju = upper.J
            Jl = lower.J
            Jdim_u = int(np.rint(2*Ju+1))
            Jdim_l = int(np.rint(2*Jl+1))
            l_dim = Idim*Jdim_l
            u_dim = Idim*Jdim_u

            subspace = np.r_[ion.slice(lower), ion.slice(upper)]
            num_states = len(subspace)

            for l_ind in list(subspace[:l_dim]):
                for u_ind in list(subspace[l_dim:]):
                    Fl = ion.F[l_ind]
                    Fu = ion.F[u_ind]
                    Ml = ion.M[l_ind]
                    Mu = ion.M[u_ind]
                    q = Ml - Mu
                    if q not in [-1, 1, 0]:
                        continue

                    spont[l_ind, u_ind] = A*(
                        (2*Ju+1)
                        *(2*Fl+1)
                        *(2*Fu+1)
                        *(wigner_3j(Fl, 1, Fu, -Ml, q, Mu))**2
                        *(wigner_6j(Jl, I, Fl, Fu, 1, Ju)**2))

        scale = np.max(np.max(np.abs(spont)))
        eps = np.max(np.max(np.abs(spont-spont_rates)))
        spont = spont[np.ix_(subspace, subspace)]
        self.assertTrue(eps/scale < 1e-4)

    def test_HF(self):
        """ Check that, in the high-field, our spontaneous scattering rates
        match a more direct calculation. """
        ion = Ca43(1000)
        rates = Rates(ion)
        Idim = int(np.rint(2*ion.I+1))
        spont_rates = rates.get_spont()
        spont = np.zeros((ion.num_states, ion.num_states))

        for _, transition in ion.transitions.items():
            A = transition.A
            lower = transition.lower
            upper = transition.upper
            Ju = upper.J
            Jl = lower.J
            Jdim_u = int(np.rint(2*Ju+1))
            Jdim_l = int(np.rint(2*Jl+1))
            l_dim = Idim*Jdim_l
            u_dim = Idim*Jdim_u

            subspace = np.r_[ion.slice(lower), ion.slice(upper)]
            num_states = len(subspace)

            for l_ind in list(subspace[:l_dim]):
                for u_ind in list(subspace[l_dim:]):
                    if ion.MI[l_ind] != ion.MI[u_ind]:
                        continue
                    M_l = ion.MJ[l_ind]
                    M_u = ion.MJ[u_ind]
                    q = M_l - M_u
                    if q not in [-1, 1, 0]:
                        continue
                    spont[l_ind, u_ind] = A*(2*Ju+1)*(
                        wigner_3j(Jl, 1, Ju, -M_l, q, M_u))**2
        scale = np.max(np.max(np.abs(spont)))
        eps = np.max(np.max(np.abs(spont-spont_rates)))
        self.assertTrue(eps/scale < 1e-8)


if __name__ == '__main__':
    unittest.main()
