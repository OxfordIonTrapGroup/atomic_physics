import numpy as np
import scipy.optimize as opt
import scipy.constants as consts
import ion_phys.common as ip

def transition_freq(B, atom, level, lower, upper):
    """ Returns the frequency of a transition between two states in the same
    level at a given magnetic field.

    :param B: the field
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :return: the transition frequency (Hz)
    """
    if isinstance(B, np.ndarray):
        B = B[0]

    ip.init(atom, B)
    E = level["E"]/consts.h
    M = level["M"]
    F = level["F"]
    return (E[np.logical_and(F == upper[0], M == upper[1])]
            - E[np.logical_and(F == lower[0], M == lower[1])])[0]

def df_dB(B, atom, level, lower, upper, eps=1e-4):
    """ Returns the field-sensitivity (Hz/T) of a transition between two states
    in the same level at a given magnetic field.

    :param B: the field
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's field sensitivity (Hz/T)
    """
    return (transition_freq(B+eps, atom, level, lower, upper)
            - transition_freq(B, atom, level, lower, upper))/eps

def d2f_dB2(B, atom, level, lower, upper, eps=1e-4):
    """ Returns the second-order field-sensitivity (Hz/T^2) of a transition
    between two states in the same level at a given magnetic field.

    :param B: the field
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's second-order field sensitivity (Hz/T^2)
    """
    return (df_dB(B+eps, atom, level, lower, upper)
            - df_dB(B, atom, level, lower, upper))/eps

def field_insensitive_point(atom, level, lower, upper, B_min=1e-3, B_max=1e-1):
    """ Returns the magnetic field at which the frequency of a transition
    between two states in the same level becomes first-order field independent.

    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param B_min: minimum magnetic field used in numerical minimization (T)
    :param B_max: maximum magnetic field used in numerical maximization (T)
    :return: the field-independent point (T) or None if none found
    """
    res = opt.root(df_dB, x0=1e-8, args=(atom, level, lower, upper),
                   options={'xtol': 1e-4, 'eps': 1e-7})

    return res.x[0] if res.success else None
