import numpy as np


def AngularMomentumRaisingOp(j: float) -> np.ndarray:
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


def AngularMomentumLoweringOp(j: float) -> np.ndarray:
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
    return AngularMomentumRaisingOp(j).T


def AngularMomentumProjectionOp(j: float) -> np.ndarray:
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


def expectation_value(state_vectors: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Calculates the expectation value of an operator for a given set of states in a
    given basis.

    :param state_vectors: array of shape ``(num_basis_states, num_states)`` where
                    ``num_basis_states`` is the number of states in the basis that ``operator`` is
                    represented in and ``num_states`` is the number of states to calculate the
                    expectation value for. The vector ``state_vector[:, state_index]`` should give
                    the representation of state ``state_index`` in the basis.
    :param operator: operator represented in the basis given by ``state_vectors``.
    :return: a vector, giving the expectation value of ``operator`` for each state in
                    ``state_vectors``.
    """
    return np.diag(state_vectors.conj().T @ operator @ state_vectors)
