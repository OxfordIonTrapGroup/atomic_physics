""" 43Ca+

References:
[1] - F. Arbes, et al., Zeitschrift fur Physik D: Atoms, Molecules and
  Clusters, 31, 27 (1994)
[2] - G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
[3] - T. P. Harty, et al. Phys. Rev. Lett 113, 220501 (2014)
[4] - W.  Nortershauser, et al., The European Physical Journal D, 2 (1998)
[5] - J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
"""

import scipy.constants as consts
from ion_phys.common import Level

atom = {}
atom["I"] = 7/2

atom["levels"] = {
    Level(n=4, L=0, S=1/2, J=1/2): {
        "Ahfs": -3225.60828640e6 * consts.h / 4,  # [1]
        "gJ": 2.00225664,  # [2]
        "gI": (2 / 7) * -1.315348  # [3]
    },
    Level(n=4, L=1, S=1/2, J=1/2): {
        "Ahfs": -145.4e6 * consts.h,  # [4]
        "gI": (2 / 7) * -1.315348  # [3]
    },
    Level(n=4, L=1, S=1/2, J=3/2): {
        "Ahfs": -31.4e6 * consts.h,  # [4]
        "Bhfs": -6.9 * consts.h,  # [4]
        "gI": (2 / 7) * -1.315348  # [3]
    },
    Level(n=3, L=2, S=1/2, J=3/2): {
        "Ahfs": -47.3e6 * consts.h,  # [4]
        "Bhfs": -3.7 * consts.h,  # [4]
        "gI": (2 / 7) * -1.315348  # [3]
    },
    Level(n=3, L=2, S=1/2, J=5/2): {
        "Ahfs": -3.8931e6 * consts.h,  # [5]
        "Bhfs": 4.241 * consts.h,  # [5]
        "gI": (2 / 7) * -1.315348  # [3]
    }
}

atom["transitions"] = {
    "397": {
        "lower": Level(n=4, L=0, S=1/2, J=1/2),
        "upper": Level(n=4, L=1, S=1/2, J=1/2),
        "A": 132e6,  # [?]
        "f0": 755.2227662e12  # [?]
    }
}
