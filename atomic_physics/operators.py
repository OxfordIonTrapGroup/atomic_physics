import numpy as np


def Jp(j: float) -> np.ndarray:
    r"""Angular momentum raising operator (:math:`J_+`) represented in the basis of
    angular momentum eigenstates.

    Basis states are labelled in order of increasing :math:`M_J`.

    The returned operator is defined so that:
    
    .. math::

      J_{+\left(n, m\right)} &:= \left<M_J=n|J_+|M_J=m\right>\\
      & = \delta\left(n,m+1\right)\sqrt{(J-m)(J+m+1)}

    This operator is related to the spherical basis operator :math:`J_{+1}` by:

    .. math::

      J_{+1} = -\frac{1}{\sqrt{2}} J_+

    """
    Mj = np.arange(-j, j)
    return np.diag(np.sqrt((j - Mj) * (j + Mj + 1)), -1)


def Jm(j: float) -> np.ndarray:
    r"""Angular momentum lowering operator (:math:`J_-`) represented in the basis of
    angular momentum eigenstates.

    Basis states are labelled in order of increasing :math:`M_J`.

    The returned operator is defined so that:
    
    .. math::

      J_{-\left(n, m\right)} &:= \left<M_J=n|J_-|M_J=m\right>\\
      & = \delta\left(n,m-1\right)\sqrt{(J+m)(J-m+1)}

    This operator is related to the spherical basis operator :math:`J_{-1}` by:

    .. math::

      J_{-1} = +\frac{1}{\sqrt{2}} J_-

    """
    return Jp(j).T


def Jz(j: float) -> np.ndarray:
    r"""Angular momentum projection operation represented in the basis of
    angular momentum eigenstates.

    Basis states are labelled in order of increasing :math:`M_J`.

    The returned operator is defined so that:

    .. math::

      J_{z\left(n, m\right)} &:= \left<M_J=n|J_z|M_J=m\right>\\
      & = m\delta\left(n,m\right)
    """
    Mj = np.arange(-j, j + 1)
    return np.diag(Mj)
