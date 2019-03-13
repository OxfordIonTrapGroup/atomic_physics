

from ion_phys.atoms.ca_43_p import atom
import ion_phys.common as ip
import numpy as np
import scipy.constants as sc
import scipy.optimize as so

muB = sc.physical_constants['Bohr magneton'][0]
muN = sc.physical_constants['nuclear magneton'][0]
amu = sc.physical_constants['unified atomic mass unit'][0]

level = atom["levels"][ip.Level(n=4, L=0, S=1/2, J=1/2)]


def local_gradient(B, F1, M1, F2, M2):
    '''Determines the local gradient of the transition frequency between
    two hyperfine manifolds as a function of magnetic field'''

    ip.init(atom, B + 1E-7)
    E = level["E"]/(1E9*sc.h)
    M = level["M"]
    F = level["F"]
    trans_freq_plus = (E[np.logical_and(M == M1, F == F1)]
                       - E[np.logical_and(M == M2, F == F2)])

    ip.init(atom, B - 1E-7)
    E = level["E"]/(1E9*sc.h)
    M = level["M"]
    F = level["F"]
    trans_freq_minus = (E[np.logical_and(M == M1, F == F1)]
                        - E[np.logical_and(M == M2, F == F2)])

    return abs((trans_freq_plus - trans_freq_minus)/2E-7)


def clock_qubit_field(F1, F2):
    '''Calculates the magnetic field at which the transition frequency between
    different magnetic sublevels is first order independent to applied
    magnetic field. Prints a list of all possible transitions and their
    respective magic field'''

    M1 = np.arange(-F1, F1+1)
    M2 = np.arange(-F2, F2+1)

    lower_field = 5         # Lower limit of B-field search in Gauss
    upper_field = 1000      # Upper limit of B-field search in Gauss

    for m_init in M1:
        for m_final in M2:
            if abs(m_final - m_init) > 1:
                continue
            else:
                res = so.minimize_scalar(local_gradient,
                                         bounds=(1e-4*lower_field,
                                                 1e-4*upper_field),
                                         args=(F1, m_init, F2, m_final),
                                         method='bounded',
                                         options={'xatol': 1E-10,
                                                  'maxiter': 500})
                magic_field = float(res.x*1e4)
                if magic_field < lower_field + 1 or magic_field > upper_field-1:
                    print('{} --> {}: No magic field'.format(m_init, m_final))
                else:
                    print('{} --> {}: Magic field: {:.2f} G'.
                          format(m_init, m_final, magic_field))


if __name__ == '__main__':
    clock_qubit_field(4, 3)
