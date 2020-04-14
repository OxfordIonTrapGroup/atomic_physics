import numpy as np
from .wigner import wigner3j


class Rates:
    def __init__(self, ion):
        self.ion = ion

        self.Gamma = None  # Gamma[i, j] rate of decay to state i from state j
        if ion.Gamma is None:
            ion.calc_Scattering()

    def get_spont(self):
        """ Returns the spontaneous emission matrix. """
        # TODO: express the spontaneous rates in terms of the multipole matrix
        # elements and move that calculation into Ion
        ion = self.ion
        self.Gamma = np.zeros((self.ion.num_states, self.ion.num_states))
        Idim = np.rint(2.0*ion.I+1).astype(int)
        for _, transition in ion.transitions.items():
            A = transition.A
            upper = transition.upper
            lower = transition.lower
            Ju = upper.J
            Jl = lower.J
            Mu = np.arange(-Ju, Ju+1)
            Ml = np.arange(-Jl, Jl+1)
            Jdim_u = int(np.rint(2*Ju+1))
            Jdim_l = int(np.rint(2*Jl+1))
            Jdim = Jdim_u + Jdim_l

            order = Ju-Jl
            if order > 1:
                print("skipping {}".format(order))
                continue

            # calculate scattering rates in the high-field basis so we can
            # forget about nuclear spin
            Gamma_hf = np.zeros((Jdim, Jdim))
            for ind_u in range(Jdim_u):
                # q = Ml - Mu
                for q in [-1, 0, 1]:
                    if abs(Mu[ind_u] + q) > Jl:
                        continue
                    ind_l = np.argwhere(Ml == Mu[ind_u]+q)
                    sign = (-1)**(-Mu[ind_u]+Jl+1)
                    Gamma_hf[ind_l, ind_u+Jdim_l] = wigner3j(
                        Jl, 1, Ju, -(Mu[ind_u]+q), q, Mu[ind_u])*sign
            Gamma_hf *= np.sqrt(A*(2*Ju+1))

            # introduce the (still decoupled) nuclear spin
            Gamma_hf = np.kron(Gamma_hf, np.identity(Idim))

            # now couple...
            subspace = np.r_[ion.slice(lower), ion.slice(upper)]
            subspace = np.ix_(subspace, subspace)
            V = ion.V[subspace]
            self.Gamma[subspace] = np.power(np.abs((V.T)@Gamma_hf@(V)), 2)

        GammaJ = sum(self.Gamma, 0)
        spont = np.copy(self.Gamma)
        for ii in range(self.Gamma.shape[0]):
            spont[ii, ii] = -GammaJ[ii]
        return spont

    def get_stim(self, lasers):
        """ Returns the stimulated emission matrix for a list of lasers. """
        ## WIP ##
        ion = self.ion
        stim = np.zeros((self.ion.num_states, self.ion.num_states))
        for transition in self.ion.transitions.keys():
            _lasers = [laser for laser in lasers
                       if laser.transition == transition]
            if _lasers == []:
                continue

            lower = ion.transitions[transition].lower
            upper = ion.transitions[transition].upper
            subspace = np.r_[ion.slice(lower), ion.slice(upper)]
            subspace = np.ix_(subspace, subspace)
            # Gamma = self.Gamma[subspace]
            # GammaJ2 = np.power(np.sum(Gamma, axis=0), 2)  # total decay rate
            for q in [-1, 0, 1]:
                q_lasers = [laser for laser in _lasers if laser.q == q]
                if q_lasers == []:
                    continue

                # print(transition, q, Gamma)
                # to do: depletion!
        return stim

    def get_tranitions(self, lasers):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)
