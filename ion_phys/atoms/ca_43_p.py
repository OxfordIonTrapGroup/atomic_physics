""" 43Ca+

References:
[1] - F. Arbes, et al., Zeitschrift fur Physik D: Atoms, Molecules and
  Clusters, 31, 27 (1994)
[2] - G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
[3] - T. P. Harty, et al. Phys. Rev. Lett 113, 220501 (2014)
"""

import scipy.constants as consts
from ion_phys.common import level

atom = {}
atom["I"] = 7/2

atom["levels"] = {
    level(n=4, L=0, S=1/2, J=1/2): {
        "Ahfs": -3225.60828640e6 * consts.h / 4,  # [1]
        "gJ": 2.00225664,  # [2]
        "gI": (2 / 7) * -1.315348  # [3]
    }
}

atom["transitions"] = {
    "397": {
        "lower": level(n=4, L=0, S=1/2, J=1/2),
        "upper": level(n=4, L=1, S=1/2, J=1/2),
        "A": 132e6,  # [?]
        "f0": 755.2227662e12  # [?]
    }
}
