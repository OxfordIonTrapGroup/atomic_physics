import numpy as np


def Jp(j):
    """ Angular momentum raising operator (J+) represented in the basis of
    angular momentum eigenstates.

    Basis states are labelled in order of increasing Mj.

    The returned operator is defined so that:
      Jp[Mi,Mj] := <Mi|Jp|Mj> = sqrt((J-Mj)(J+Mj+1))*delta(Mi,Mj+1)

    This operator is related to the spherical basis operator J+1 by:
      J+1 = -1/(sqrt(2)) J+
    """
    Mj = np.arange(-j, j)
    return np.diag(np.sqrt((j-Mj)*(j+Mj+1)), -1)


def Jm(j):
    """ Angular momentum lowering operation (J-).

    See Jp.

    This operator is related to the spherical basis operator J-1 by:
      J-1 = +1/(sqrt(2)) J-
    """
    return Jp(j).T


def Jz(j):
    """ Angular momentum projection operation represented in the basis of
    angular momentum eigenstates.

    Basis states are labelled in order of increasing Mj.

    The returned operator is defined so that:
      Jp[Mi,Mj] := <Mi|Jz|Mj> = Mi*delta(Mi,Mj+1)
    """
    Mj = np.arange(-j, j+1)
    return np.diag(Mj)
