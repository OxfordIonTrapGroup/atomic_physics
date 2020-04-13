import numpy as np


class Rates:
    def __init__(self, ion):
        self.ion = ion

        if ion.Gamma is None:
            ion.calc_Scattering()

    def get_spont(self):
        """ Returns the spontaneous emission matrix. """
        return np.zeros((self.ion.num_states, self.ion.num_states))

    def get_stim(self, lasers):
        """ Returns the stimulated emission matrix for a list of lasers. """
        for laser in lasers:
            pass
        return np.zeros((self.ion.num_states, self.ion.num_states))

    def get_tranitions(self, lasers):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)
