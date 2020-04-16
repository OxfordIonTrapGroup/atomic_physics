import numpy as np


class Rates:
    def __init__(self, ion):
        self.ion = ion

        if ion.ePole is None:
            ion.calc_Epole()

    def get_spont(self):
        """ Returns the spontaneous emission matrix. """
        Gamma = np.power(np.abs(self.ion.ePole), 2)
        for ii in range(Gamma.shape[0]):
            Gamma[ii, ii] = -self.ion.GammaJ[ii]
        return Gamma

    def get_stim(self, lasers):
        """ Returns the stimulated emission matrix for a list of lasers. """
        Gamma = np.power(np.abs(self.ion.ePole), 2)
        GammaJ = self.ion.GammaJ
        stim = np.zeros(Gamma.shape)

        for transition in self.ion.transitions.keys():
            _lasers = [laser for laser in lasers
                       if laser.transition == transition]
            if _lasers == []:
                continue

            lower = self.ion.transitions[transition].lower
            upper = self.ion.transitions[transition].upper
            lower_states = self.ion.slice(lower)
            upper_states = self.ion.slice(upper)
            n_lower = self.ion.levels[lower]._num_states
            n_upper = self.ion.levels[upper]._num_states

            dJ = upper.J-lower.J
            dL = upper.L - lower.L
            if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                order = 1
            elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                order = 2
            else:
                raise ValueError("Unsupported transition order {}"
                                 .format(order))

            Mu = self.ion.M[upper_states]
            Ml = self.ion.M[lower_states]
            Mu = np.repeat(Mu, n_lower).reshape(n_upper, n_lower).T
            Ml = np.repeat(Ml, n_upper).reshape(n_lower, n_upper)

            # Transition detunings
            El = self.ion.E[lower_states]
            Eu = self.ion.E[upper_states]
            El = np.repeat(El, n_upper).reshape(n_lower, n_upper)
            Eu = np.repeat(Eu, n_lower).reshape(n_upper, n_lower).T
            delta_lu = Eu - El

            # Total scattering rate out of each state
            GammaJ = GammaJ[upper_states]
            GammaJ = np.repeat(GammaJ, n_lower).reshape(n_upper, n_lower).T
            GammaJ2 = np.power(GammaJ, 2)

            Gamma = Gamma[lower_states, upper_states]
            R = np.zeros((n_lower, n_upper))
            for q in range(-order, order+1):
                Q = np.zeros((n_lower, n_upper))
                # q := Mu - Ml
                Q[Ml == (Mu-q)] = 1
                for laser in [laser for laser in _lasers if laser.q == q]:
                    delta = delta_lu - laser.delta
                    I = laser.I
                    R += GammaJ2/(4*np.power(delta, 2) + GammaJ2)*I*(Q*Gamma)
                assert (R >= 0).all()

            stim[lower_states, upper_states] = R
            stim[upper_states, lower_states] = R.T

        stim_j = np.sum(stim, 0)
        for ii in range(self.ion.num_states):
            stim[ii, ii] = -stim_j[ii]
        return stim

    def get_transitions(self, lasers):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)
