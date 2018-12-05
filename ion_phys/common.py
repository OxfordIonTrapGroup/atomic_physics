import numpy as np
from collections import namedtuple
import scipy.constants as consts

from ion_phys import operators


level = namedtuple("level", "n,L,S,J")

_uB = consts.physical_constants["Bohr magneton"][0]
_uN = consts.physical_constants["nuclear magneton"][0]


def init(atom, B):
    """ Calculate atomic properties at a given B-field (Tesla).

    We add the following fields to each level dictionary:
      M - Magnetic quantum number for each state
      E - Energy of each state relative to the level's centre of gravity,
        sorted in order of increasing energy.
      V - V[:, i] is the state with energy E[i], represented in the basis of
        high-field (MI, MJ) energy eigenstates.
      MI, MJ - high-field energy eigenstate basis vectors
    """
    atom["B"] = B
    I = atom["I"] = atom.get("I", 0)

    I_dim = np.rint(2.0*atom["I"]+1).astype(int)

    for level, data in atom["levels"].items():

        gJ = data["gJ"] = data.get("gJ", Lande_g(level))

        J_dim = np.rint(2.0*level.J+1).astype(int)

        Jp = np.kron(operators.Jp(level.J), np.identity(I_dim))
        Jm = np.kron(operators.Jm(level.J), np.identity(I_dim))
        Jz = np.kron(operators.Jz(level.J), np.identity(I_dim))

        Ip = np.kron(np.identity(J_dim), operators.Jp(I))
        Im = np.kron(np.identity(J_dim), operators.Jm(I))
        Iz = np.kron(np.identity(J_dim), operators.Jz(I))

        H = gJ*_uB*B*Jz
        if atom["I"] != 0:
            A = data["Ahfs"]
            gI = data["gI"]
            H += A*(Iz@Jz + (1/2)*(Ip@Jm + Im@Jp)) - gI*_uN*B*Iz

        E, V = np.linalg.eig(H)

        inds = np.argsort(E)
        data["E"] = E[inds]
        data["V"] = V[:, inds]

        data["MI"] = np.kron(np.ones(J_dim), np.arange(-I, I + 1))
        data["MJ"] = np.kron(np.arange(-level.J, level.J + 1), np.ones(I_dim))

        M = data["MI"] + data["MJ"]
        data["M"] = M[np.argmax(np.abs(data["V"]), 0)]


def Lande_g(level):
    gL = 1
    gS = -consts.physical_constants["electron g factor"]

    S = level.S
    J = level.J
    L = level.L

    gJ = gL*(J*(J+1) - S*(S+1) + L*(L+1)) / (2*J*(J+1)) \
        + gS*(J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    return gJ


def calc_m1(atom):
    """
    Calculates the matrix elements for M1 transitions within each level.

    The matrix elements for each level are stored as the matrix
    atom["levels"][level].R_m1 defined so that:
      - R_m1[i, j] := (-1)**(q+1)<i|u_q|j>
      - q := Mi - Mj = (-1, 0, 1)
      - u_q is the qth component of the magnetic dipole operator in spherical
        coordinates.

    NB with this definition, the Rabi frequency is given by:
      - hbar * W = B_-q * R
      - t_pi = pi/W
      - where B_-q is the -qth component of the magnetic field in spherical
        coordinates.
    """
    I = atom["I"]
    I_dim = np.rint(2.0*I+1).astype(int)
    eyeI = np.identity(I_dim)

    for level, data in atom["levels"].items():

        J_dim = np.rint(2.0*level.J+1).astype(int)
        dim = J_dim*I_dim
        eyeJ = np.identity(J_dim)

        # magnetic dipole operator in spherical coordinates
        Jp = np.kron((-1/np.sqrt(2))*operators.Jp(level.J), eyeI)
        Jm = np.kron((+1/np.sqrt(2))*operators.Jm(level.J), eyeI)
        Jz = np.kron(operators.Jz(level.J), eyeI)

        Ip = np.kron(eyeJ, (-1/np.sqrt(2))*operators.Jp(I))
        Im = np.kron(eyeJ, (+1/np.sqrt(2))*operators.Jm(I))
        Iz = np.kron(eyeJ, operators.Jz(I))

        up = (-data["gJ"]*_uB*Jp + data["gI"]*_uN*Ip)
        um = (-data["gJ"]*_uB*Jm + data["gI"]*_uN*Im)
        uz = (-data["gJ"]*_uB*Jz + data["gI"]*_uN*Iz)

        u = [um, uz, up]

        Mj = np.tile(data["M"], (dim, 1))
        Mi = Mj.T
        Q = (Mi - Mj)

        valid = (np.abs(Q) <= 1)
        valid[np.diag_indices(dim)] = False

        data["R_m1"] = np.zeros((dim, dim))
        for transition in np.nditer(np.nonzero(valid)):
            i = transition[0]
            j = transition[1]
            q = np.rint(Q[i, j]).astype(int)

            psi_i = data["V"][:, i]
            psi_j = data["V"][:, j]

            data["R_m1"][i, j] = ((-1)**(q+1)) * psi_i.T@u[q+1]@psi_j
