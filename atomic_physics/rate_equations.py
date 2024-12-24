import numpy as np

from atomic_physics.common import Atom, Laser, Transition


class Rates:
    def __init__(self, atom: Atom):
        self.atom = atom

        if atom.ePole is None:
            atom.calc_Epole()

    def get_spont(self):
        """Returns the spontaneous emission matrix."""
        Gamma = np.power(np.abs(self.atom.ePole), 2)
        for ii in range(Gamma.shape[0]):
            Gamma[ii, ii] = -self.atom.GammaJ[ii]
        return Gamma

    def get_stim(self, lasers: list[Laser]):
        """Returns the stimulated emission matrix for a list of lasers."""
        Gamma = np.power(np.abs(self.atom.ePole), 2)
        GammaJ = self.atom.GammaJ
        stim = np.zeros(Gamma.shape)

        for transition in self.atom.transitions.keys():
            _lasers = [laser for laser in lasers if laser.transition == transition]
            if _lasers == []:
                continue

            lower = self.atom.transitions[transition].lower
            upper = self.atom.transitions[transition].upper
            lower_states = self.atom.get_slice(lower)
            upper_states = self.atom.get_slice(upper)
            n_lower = self.atom.level_states[lower].num_states
            n_upper = self.atom.level_states[upper].num_states

            dJ = upper.J - lower.J
            dL = upper.L - lower.L
            if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                order = 1
            elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                order = 2
            else:
                raise ValueError(
                    "Unsupported transition order. \n"
                    "Only 1st and 2nd order transitions are "
                    "supported. [abs(dL) & abs(dJ) <2]\n"
                    "Got dJ={} and dL={}".format(dJ, dL)
                )

            Mu = self.atom.M[upper_states]
            Ml = self.atom.M[lower_states]
            Mu = np.repeat(Mu, n_lower).reshape(n_upper, n_lower).T
            Ml = np.repeat(Ml, n_upper).reshape(n_lower, n_upper)

            # Transition detunings
            El = self.atom.E[lower_states]
            Eu = self.atom.E[upper_states]
            El = np.repeat(El, n_upper).reshape(n_lower, n_upper)
            Eu = np.repeat(Eu, n_lower).reshape(n_upper, n_lower).T
            delta_lu = Eu - El

            # Total scattering rate out of each state
            GammaJ_subs = GammaJ[upper_states]
            GammaJ_subs = np.repeat(GammaJ_subs, n_lower).reshape(n_upper, n_lower).T
            GammaJ2 = np.power(GammaJ_subs, 2)

            Gamma_subs = Gamma[lower_states, upper_states]
            R = np.zeros((n_lower, n_upper))
            for q in range(-order, order + 1):
                Q = np.zeros((n_lower, n_upper))
                # q := Mu - Ml
                Q[Ml == (Mu - q)] = 1
                for laser in [laser for laser in _lasers if laser.q == q]:
                    delta = delta_lu - laser.delta
                    I = laser.I
                    R += (
                        GammaJ2
                        / (4 * np.power(delta, 2) + GammaJ2)
                        * I
                        * (Q * Gamma_subs)
                    )
                assert (R >= 0).all()

            stim[lower_states, upper_states] = R
            stim[upper_states, lower_states] = R.T

        stim_j = np.sum(stim, 0)
        for ii in range(self.atom.num_states):
            stim[ii, ii] = -stim_j[ii]
        return stim

    def get_transitions(self, lasers: list[Laser]):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)

    def steady_state(
        self,
        *,
        trans: Transition | None = None,
        lasers: list[Laser] | None = None,
    ):
        """Returns the steady-state vector for *either* a transitions matrix
        or a list of lasers.
        :param trans: transitions matrix to solve for
        :param lasers: laser list to solve for
        :returns: state vector
        """
        if sum([x is not None for x in (trans, lasers)]) != 1:
            raise ValueError("Exactly one of trans and lasers must not be None")

        if lasers is not None:
            trans = self.get_transitions(lasers)
        else:
            trans = np.copy(trans)  # don't overwrite their matrix!

        trans[0, :] = 1
        Vi = np.zeros((trans.shape[0], 1))
        Vi[0] = 1
        Vf, _, _, _ = np.linalg.lstsq(trans, Vi, rcond=None)
        return Vf
