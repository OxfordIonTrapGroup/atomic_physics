import numpy as np

from atomic_physics.core import Atom, Laser


class Rates:
    """Rate equations calculations.

    See the :ref:`rates` section of the documentation for details.

    Example usage:

    .. testcode::

       # Electron shelving simulation in 43Ca+
       import matplotlib.pyplot as plt
       import numpy as np
       from scipy.linalg import expm

       from atomic_physics.core import Laser
       from atomic_physics.ions import ca43
       from atomic_physics.rate_equations import Rates

       t_ax = np.linspace(0, 100e-6, 100)  # Scan the duration of the "shelving" pulse
       intensity = 0.02  # 393 intensity

       ion = ca43.Ca43(magnetic_field=146e-4)

       # Ion starts in the F=4, M=+4 "stretched" state within the 4S1/2 ground-level
       stretch = ion.get_state_for_F(ca43.ground_level, F=4, M_F=+4)
       Vi = np.zeros((ion.num_states, 1))
       Vi[stretch] = 1

       # Tune the 393nm laser to resonance with the
       # 4S1/2(F=4, M_F=+4) <> 4P3/2(F=5, M_F=+5) transition
       detuning = ion.get_transition_frequency_for_states(
           (stretch, ion.get_state_for_F(ca43.P32, F=5, M_F=+5))
       )
       lasers = (
           Laser("393", polarization=+1, intensity=intensity, detuning=detuning),
       )  # resonant 393 sigma+

       rates = Rates(ion)
       transitions = rates.get_transitions_matrix(lasers)

       shelved = np.zeros(len(t_ax))  # Population in the 3D5/2 level at the end
       for idx, t in np.ndenumerate(t_ax):
           Vf = expm(transitions * t) @ Vi  # NB use of matrix operations here!
           shelved[idx] = np.sum(Vf[ion.get_slice_for_level(ca43.shelf)])

       plt.plot(t_ax * 1e6, shelved)
       plt.ylabel("Shelved Population")
       plt.xlabel("Shelving time (us)")
       plt.grid()
       plt.show()

    """

    def __init__(self, atom: Atom):
        self.atom = atom

    def get_spont_matrix(self) -> np.ndarray:
        """Returns the spontaneous emission matrix."""
        scattering_rates = np.abs(self.atom.get_electric_multipoles()) ** 2
        total_rates = np.sum(scattering_rates, 0)

        for ii in range(scattering_rates.shape[0]):
            scattering_rates[ii, ii] = -total_rates[ii]

        return scattering_rates

    def get_stim_matrix(self, lasers: tuple[Laser, ...]) -> np.ndarray:
        """Returns the stimulated emission matrix for a set of lasers."""
        scattering_rates = np.abs(self.atom.get_electric_multipoles()) ** 2
        total_rates = np.sum(scattering_rates, 0)

        stim = np.zeros(scattering_rates.shape)

        for transition in self.atom.transitions.keys():
            _lasers = [laser for laser in lasers if laser.transition == transition]
            if _lasers == []:
                continue

            lower = self.atom.transitions[transition].lower
            upper = self.atom.transitions[transition].upper
            lower_states = self.atom.get_slice_for_level(lower)
            upper_states = self.atom.get_slice_for_level(upper)
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
            El = self.atom.state_energies[lower_states]
            Eu = self.atom.state_energies[upper_states]
            El = np.repeat(El, n_upper).reshape(n_lower, n_upper)
            Eu = np.repeat(Eu, n_lower).reshape(n_upper, n_lower).T
            delta_lu = Eu - El

            # Total scattering rate out of each state
            total_rates_subs = total_rates[upper_states]
            total_rates_subs = (
                np.repeat(total_rates_subs, n_lower).reshape(n_upper, n_lower).T
            )
            total_rates_2 = np.power(total_rates_subs, 2)

            scattering_rates_subs = scattering_rates[lower_states, upper_states]
            R = np.zeros((n_lower, n_upper))
            for q in range(-order, order + 1):
                Q = np.zeros((n_lower, n_upper))
                # q := Mu - Ml
                Q[Ml == (Mu - q)] = 1
                for laser in [laser for laser in _lasers if laser.polarization == q]:
                    delta = delta_lu - laser.detuning
                    R += (
                        total_rates_2
                        / (4 * np.power(delta, 2) + total_rates_2)
                        * laser.intensity
                        * (Q * scattering_rates_subs)
                    )
                assert (R >= 0).all()

            stim[lower_states, upper_states] = R
            stim[upper_states, lower_states] = R.T

        stim_j = np.sum(stim, 0)
        for ii in range(self.atom.num_states):
            stim[ii, ii] = -stim_j[ii]
        return stim

    def get_transitions_matrix(self, lasers: tuple[Laser, ...]) -> np.ndarray:
        """Returns the complete (spontaneous + stimulated emissions) transitions matrix
        for a given set of lasers.
        """
        return self.get_spont_matrix() + self.get_stim_matrix(lasers)

    def get_steady_state_populations(self, transitions: np.ndarray) -> np.ndarray:
        """Returns the steady-state population vector for a given transitions matrix.

        :param transitions: transitions matrix.
        :returns: steady-state population vector.
        """
        t_pr = np.copy(transitions)
        t_pr[0, :] = 1
        b = np.zeros(self.atom.num_states)
        b[0] = 1
        Vf, _, _, _ = np.linalg.lstsq(t_pr, b, rcond=None)
        return Vf
