from sympy.physics.wigner import wigner_3j as _wigner_3j
from sympy.physics.wigner import wigner_6j as _wigner_6j


def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    """Temporary work around for https://github.com/sympy/sympy/pull/27288"""
    return _wigner_3j(*map(float, (j_1, j_2, j_3, m_1, m_2, m_3)))


def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    """Temporary work around for https://github.com/sympy/sympy/pull/27288"""
    return _wigner_6j(*map(float, (j_1, j_2, j_3, j_4, j_5, j_6)), prec)
