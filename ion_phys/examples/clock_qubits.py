import ion_phys.common as ip
from ion_phys.atoms.ca_43_p import atom
from ion_phys.utils import field_insensitive_point, transition_freq, d2f_dB2


if __name__ == '__main__':
    # all seems about correct (e.g. agrees with TPH thesis) but expect some
    # numerical inaccuracy, particularly around the second-order field
    # sensitivities
    level = atom["levels"][ip.Level(n=4, L=0, S=1/2, J=1/2)]
    print("Filed-independent points:")
    for M3 in range(-3, +3 + 1):
        for q in [-1, 0, 1]:
            B0 = field_insensitive_point(atom, level, (4, M3-q), (3, M3))
            if B0 is not None:
                f0 = transition_freq(B0, atom, level, (4, M3-q), (3, M3))
                d2fdB2 = d2f_dB2(B0, atom, level, (4, M3-q), (3, M3))
                print("4, {} --> 3, {}: {:.1f} Hz @ {:.5f} mT ({:.3e} Hz/mT^2)"
                      .format(M3-q, M3, f0, B0*1e3, d2fdB2*1e-11))
            else:
                print("4, {} --> 3, {}: none found".format(M3-q, M3))

