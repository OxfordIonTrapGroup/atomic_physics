from pprint import pprint

from ion_phys.atoms.ca_43_p import atom
from ion_phys.common import init, calc_m1

init(atom, 146.0942e-4)
calc_m1(atom)
pprint(atom)
