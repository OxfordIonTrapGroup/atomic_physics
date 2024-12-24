from dataclasses import dataclass

import numpy as np
import scipy.constants as consts

import atomic_physics as ap

_uB = consts.physical_constants["Bohr magneton"][0]
_uN = consts.physical_constants["nuclear magneton"][0]


@dataclass(frozen=True)
class Level:
    """Represents a single level.

    Attributes:
        n: the level's principal quantum number
        L: the level's orbital angular momentum
        J: the level's total (spin + orbital) angular momentum
        S: the level's spin angular momentum
    """

    n: int
    L: float
    J: float
    S: float


@dataclass
class LevelData:
    """Atomic structure data about a single level.

    Attributes:
        level: the :class:`Level` this data is for.
        g_J: G factor. If ``None``, we use the Lande g factor.
        g_I: Nuclear g factor.
        Ahfs: Nuclear A coefficient.
        Bhfs: Nuclear B (quadrupole) coefficient.
    """

    level: Level
    g_J: float | None = None
    g_I: float | None = None
    Ahfs: float | None = None
    Bhfs: float | None = None

    def __post_init__(self):
        if self.g_J is None:
            self.g_J = ap.utils.Lande_g(self.level)


@dataclass(frozen=True)
class LevelStates:
    """Stores information about the states within a level.

    Attributes:
        freq: frequency of the centre-of-gravity transition from the ground-level
            to this level.
        start_ind: index into the state vector of the lowest-lying state within
            this level.
        stop_ind: index into the state vector of the highest-lying state within
            this level.
        num_states: the number of states within the level.
    """

    freq: float
    start_index: int
    stop_index: int
    num_states: int


@dataclass(frozen=True)
class Transition:
    """Represents a transition between a pair of states.

    Attributes:
        lower: the :class:`Level` in the transition with lower energy
        upper:  the :class:`Level` in the transition with greater energy
        freq: the transition frequency (rad/s)
        A: the transition's Einstein A coefficient
    """

    lower: Level
    upper: Level
    freq: float
    A: float


@dataclass
class Laser:
    """Represents a laser.

    Attributes:
        transition: string giving the name of the transition driven by the laser.
        q: the laser's polarization, defined as the difference in angular momentum
            between the higher and lower energy states coupled by this laser
            (``q = M_upper - M_lower``). ``q = +1`` for σ+ light, ``q = -1`` for σ-
            light and ``q = 0`` for π light.
        I: the laser intensity (saturation intensities).
        delta: the laser's detuning (rad/s) from the transition centre of gravity,
           defined so that ``w_laser = w_transition + delta``.
    """

    transition: str
    q: int
    I: float
    delta: float


class Atom:
    """Base class for storing atomic structure data.

    Attributes:
        num_states: the number of states contained within the atom
        level_states: dictionary mapping :class:`Level`s to :class:`LevelStates`.
    """

    num_states: int
    level_states: dict[Level, LevelStates]

    def __init__(
        self,
        *,
        level_data: list[LevelData],
        transitions: dict[str, Transition],
        B: float | None = None,
        I: float = 0,
        level_filter: list[Level] | None = None,
    ):
        """
        :param B: Magnetic field (T). To change the B-field later, call
          :meth setB:
        :param I: Nuclear spin
        :param level_data: list of atomic structure for each level in the atom
        :param transitions: dictionary mapping transition name strings to
          Transition objects.
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.

        Internally, we store all information as vectors/matrices with states
        ordered in terms of increasing energies.
        """
        self.B = None
        self.I = I

        levels = {data.level: data for data in level_data}
        transitions = dict(transitions)

        if level_filter is not None:
            levels = dict(filter(lambda lev: lev[0] in level_filter, levels.items()))

        transition_filter = [
            trans
            for trans in transitions.keys()
            if transitions[trans].lower in levels.keys()
            and transitions[trans].upper in levels.keys()
        ]

        transitions = dict(
            filter(lambda trans: trans[0] in transition_filter, transitions.items())
        )

        self.levels = levels
        self.transitions = transitions

        # use transition data to order levels in terms of increasing energy
        processed_levels: dict[Level, float] = {}  # level: energy (freq units)
        unprocessed_transition_names: list[str] = []

        if len(self.levels) == 1:
            processed_levels[list(self.levels.keys())[0]] = 0.0
        else:
            unprocessed_transition_names += list(self.transitions.keys())
            transition = self.transitions[unprocessed_transition_names.pop()]
            processed_levels[transition.lower] = 0
            processed_levels[transition.upper] = transition.freq

        while unprocessed_transition_names:
            for transition_name in unprocessed_transition_names:
                transition = self.transitions[transition_name]
                if transition.lower in processed_levels:
                    freq = processed_levels[transition.lower] + transition.freq
                    processed_levels[transition.upper] = freq
                    break
                elif transition.upper in processed_levels:
                    freq = processed_levels[transition.upper] - transition.freq
                    processed_levels[transition.lower] = freq
                    break
            else:
                raise ValueError(
                    "Transition '{}' would lead to a disconnected level"
                    " structure.".format(transition_name)
                )
            unprocessed_transition_names.remove(transition_name)

        if processed_levels.keys() != self.levels.keys():
            raise ValueError("Disconnected level structure")

        sorted_levels: list[tuple[Level, float]] = sorted(
            processed_levels.items(), key=lambda x: x[1]
        )

        f0 = sorted_levels[0][1]  # ground-level frequency offset
        start_index = 0
        self.level_states: dict[Level:LevelStates] = {}
        for level, level_freq in sorted_levels:
            num_states = int(np.rint((2 * self.I + 1) * (2 * level.J + 1)))
            self.level_states[level] = LevelStates(
                freq=level_freq - f0,
                start_index=start_index,
                stop_index=start_index + num_states,
                num_states=num_states,
            )
            start_index += num_states

        self.num_states = start_index

        # ordered in terms of increasing state energies
        self.M = None  # Magnetic quantum number of each state
        self.F = None  # F for each state (valid at low field)
        self.MI = None  # MI for each state (only valid at low field)
        self.MJ = None  # MJ for each state (only valid at low field)
        self.E = None  # State energies in angular frequency units

        self.ePole = None  # Scattering amplitudes
        self.ePole_hf = None  # Scattering amplitudes in the high-field basis
        self.GammaJ = None  # Total scattering rate out of each state
        self.M1 = None  # Magnetic dipole matrix elements

        # V - V[:, i] is the state with energy E[i], represented in the basis
        #    high-field (MI, MJ) energy eigenstates.
        #  MI, MJ - high-field energy eigenstate basis vectors
        # V[:, i] are the expansion coefficients for the state E[i] in the
        # basis of high-field (MI, MJ) energy eigenstates.
        # NB the MI, MJ axes here are the most convenient to represent our
        # operators in; they are not energy ordered (like self.MI, self.MJ)
        self.V = None
        self.MIax = None
        self.MJax = None

        if B is not None:
            self.setB(B)

    def get_slice(self, level: Level):
        """Returns a slice object that selects the states within a given
        level.

        Internally, we store states in order of increasing energy. This
        provides a more convenient means of accessing the states within a
        given level.
        """
        states = self.level_states[level]
        return slice(states.start_index, states.stop_index)

    def index(
        self,
        level: Level,
        M: float,
        *,
        F: float | None = None,
        MI: float | None = None,
        MJ: float | None = None,
    ):
        """Returns the index of a state.

        If no kwargs are given, we return an array of indices of all states
        with a given M. The kwargs can be used to filter the results, for
        example, only returning the state with a given F.

        Valid kwargs: F, MI, MJ

        Internally, we store states in order of increasing energy. This
        provides a more convenient means of accessing a state.
        """
        lev = self.get_slice(level)
        Mvec = self.M[lev]
        inds = Mvec == M

        if F is not None:
            Fvec = self.F[lev]
            inds = np.logical_and(inds, Fvec == F)
        if MI is not None:
            MIvec = self.MI[lev]
            inds = np.logical_and(inds, MIvec == MI)
        if MJ is not None:
            MJvec = self.MJ[lev]
            inds = np.logical_and(inds, MJvec == MJ)

        inds = np.argwhere(inds)
        if len(inds) == 1:
            inds = inds.ravel()[0]
        return inds + self.level_states[level].start_index

    def level(self, state: int):
        """Returns the level a state lies in."""
        for level, level_states in self.level_states.items():
            if state >= level_states.start_index and state < level_states.stop_index:
                return level

        raise ValueError(f"No state with index {state}")

    def delta(self, lower: int, upper: int):
        """Returns the detuning of the transition between a pair of states
        from the overall centre of gravity of the set of transitions between
        the levels containing those states.

        If both states are in the same level, this returns the transition
        frequency.

        :param lower: index of the lower state
        :param upper: index of the upper state
        :return: the detuning (rad/s)
        """
        return self.E[upper] - self.E[lower]

    def population(self, state: np.ndarray, inds: Level | int | slice):
        """Returns the total population in a set of states.

        :param state: state vector
        :param states: set of states to sum over. This can be any of: a Level;
          a state index; or, a slice.
        """
        if isinstance(inds, Level):
            return np.sum(state[self.get_slice(inds)])
        elif not isinstance(inds, int) and not isinstance(inds, slice):
            raise TypeError("inds must be a level, slice or index")
        return np.sum(state[inds])

    def I0(self, transition: Transition):
        """Returns the saturation intensity for a transition.

        We adopt a convention whereby, for a resonantly-driven cycling
        transition, one saturation intensity gives equal stimulated and
        spontaneous transition rates.

        To do: this should be double checked against our old MatLab code...

        :param transition: the transition name
        :return: saturation intensity (W/m^2)
        """
        if self.GammaJ is None:
            self.calc_Epole()

        trans = self.transitions[transition]
        omega = trans.freq
        Gamma = self.GammaJ[self.level_states[trans.upper].start_ind]
        return consts.hbar * (omega**3) * Gamma / (6 * np.pi * (consts.c**2))

    def P0(self, transition: Transition, w0: float):
        """Returns the power needed at the focus of a Guassian beam with waist
        w0 to give an on-axis intensity of I0.

        :param transition: the transition name
        :param w0: Gaussian beam waist (1/e^2 intensity radius in m)
        :return: beam power (W)
        """
        I0 = self.I0(transition)
        return 0.5 * np.pi * (w0**2) * I0

    def setB(self, B: float):
        """Calculate atomic data at a given B-field (Tesla)."""
        self.B = B
        self.M = np.zeros(self.num_states)
        self.F = np.zeros(self.num_states)
        self.E = np.zeros(self.num_states)
        self.MI = np.zeros(self.num_states)
        self.MJ = np.zeros(self.num_states)
        self.MIax = np.zeros(self.num_states)
        self.MJax = np.zeros(self.num_states)
        self.V = np.zeros((self.num_states, self.num_states))

        I = self.I
        I_dim = np.rint(2.0 * I + 1).astype(int)

        for level, data in self.levels.items():
            J = level.J
            J_dim = np.rint(2.0 * J + 1).astype(int)

            Jp = np.kron(ap.operators.Jp(J), np.identity(I_dim))
            Jm = np.kron(ap.operators.Jm(J), np.identity(I_dim))
            Jz = np.kron(ap.operators.Jz(J), np.identity(I_dim))

            Ip = np.kron(np.identity(J_dim), ap.operators.Jp(I))
            Im = np.kron(np.identity(J_dim), ap.operators.Jm(I))
            Iz = np.kron(np.identity(J_dim), ap.operators.Jz(I))

            H = data.g_J * _uB * B * Jz
            if self.I != 0:
                gI = data.g_I
                IdotJ = Iz @ Jz + (1 / 2) * (Ip @ Jm + Im @ Jp)

                H += -gI * _uN * B * Iz
                H += data.Ahfs * IdotJ

                if J > 1 / 2 and I > 1 / 2:
                    IdotJ2 = np.linalg.matrix_power(IdotJ, 2)
                    ident = np.identity(I_dim * J_dim)
                    H += (
                        data.Bhfs
                        / (2 * I * J * (2 * I - 1) * (2 * J - 1))
                        * (
                            3 * IdotJ2
                            + (3 / 2) * IdotJ
                            - ident * I * (I + 1) * J * (J + 1)
                        )
                    )

            H /= consts.hbar  # work in angular frequency units
            level_slice = self.get_slice(level)
            E, V = np.linalg.eig(H)
            inds = np.argsort(E)
            V = V[:, inds]
            E = E[inds]

            # check that the eigensolver found the angular momentum eigenstates
            M = np.diag(V.conj().T @ (Iz + Jz) @ V)
            if max(abs(M - np.rint(2 * M) / 2)) > 1e-5:
                raise ValueError(
                    "Error finding angular momentum"
                    " eigenstates at {}T. Is the field too small"
                    " to lift the state degeneracy?".format(B)
                )

            self.E[level_slice] = E
            self.V[level_slice, level_slice] = V
            self.M[level_slice] = np.rint(2 * M) / 2
            self.MIax[level_slice] = np.kron(np.ones(J_dim), np.arange(-I, I + 1))
            self.MJax[level_slice] = np.kron(np.arange(-J, J + 1), np.ones(I_dim))
            self.MI[level_slice] = np.rint(2 * np.diag(V.conj().T @ (Iz) @ V)) / 2
            self.MJ[level_slice] = np.rint(2 * np.diag(V.conj().T @ (Jz) @ V)) / 2

            F_list = np.arange(abs(I - J), I + J + 1)
            if self.I != 0 and data.Ahfs < 0:
                F_list = F_list[::-1]

            for M in set(self.M[level_slice]):
                for Fidx, idx in np.ndenumerate(np.where(M == self.M[level_slice])):
                    self.F[level_slice][idx] = F_list[abs(M) <= F_list][Fidx[1]]

        if self.M1 is not None:
            self.calc_M1()
        if self.ePole is not None:
            self.calc_Epole()

    def calc_Epole(self):
        """Calculate the electric multi-pole matrix elements for each
        transition. Currently we only calculate dipole (E1) and Quadrupole
        (E2) matrix elements.

        Strictly speaking, we actually store scattering amplitudes (square
        root of the spontaneous scattering rates), rather than matrix elements,
        since they are somewhat more convenient to work with. The two are
        related by constants and power of the transition frequencies.

        To do: define what we mean by a matrix element, and how the amplitudes
        stored in ePole are related to the matrix elements and to the Rabi
        frequencies.

        To do: double check the signs of the amplitudes.
        """
        # We calculate the scattering amplitudes in the high-field basis where
        # we can forget about nuclear spin. It's then straightforward to
        # transform to arbitrary fields using the expansion coefficients we've
        # already calculated.
        if self.ePole_hf is None:
            self.ePole_hf = np.zeros((self.num_states, self.num_states))
            Idim = np.rint(2.0 * self.I + 1).astype(int)

            for _, transition in self.transitions.items():
                A = transition.A
                upper = transition.upper
                lower = transition.lower
                Ju = upper.J
                Jl = lower.J
                Mu = np.arange(-Ju, Ju + 1)
                Ml = np.arange(-Jl, Jl + 1)
                Jdim_u = int(np.rint(2 * Ju + 1))
                Jdim_l = int(np.rint(2 * Jl + 1))
                Jdim = Jdim_u + Jdim_l

                subspace = np.r_[self.get_slice(lower), self.get_slice(upper)]
                subspace = np.ix_(subspace, subspace)

                dJ = Ju - Jl
                dL = upper.L - lower.L
                if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                    order = 1
                elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                    order = 2
                else:
                    raise ValueError(
                        "Unsupported transition order. \n"
                        "Only 1st and 2nd order transitions are "
                        "supported. [abs(dL) & abs(dJ) <2]\n"
                        "Got dJ={} and dL={}".format(dJ, dL)
                    )

                # High-field scattering rates for this transition (ignoring I)
                ePole_hf = np.zeros((Jdim, Jdim))
                for ind_u in range(Jdim_u):
                    # q := Mu - Ml
                    for q in range(-order, order + 1):
                        if abs(Mu[ind_u] - q) > Jl:
                            continue

                        ind_l = np.argwhere(Ml == Mu[ind_u] - q)
                        sign = (-1) ** (2 * Ju + Jl - Mu[ind_u] + order)
                        ePole_hf[ind_l, ind_u + Jdim_l] = (
                            ap.wigner.wigner3j(
                                Ju, order, Jl, -Mu[ind_u], q, (Mu[ind_u] - q)
                            )
                            * sign
                        )
                ePole_hf *= np.sqrt(A * (2 * Ju + 1))

                # introduce the (still decoupled) nuclear spin
                self.ePole_hf[subspace] = np.kron(ePole_hf, np.identity(Idim))

        # now couple...
        V = self.V
        self.ePole = V.T @ self.ePole_hf @ V
        self.Gamma = np.power(np.abs(self.ePole), 2)
        self.GammaJ = np.sum(self.Gamma, 0)

    def calc_M1(self):
        """Calculates the matrix elements for M1 transitions within each
        level.

        The matrix elements, ``Rij``, are defined so that::

          * ``R[i, j] := (-1)**(q+1)<i|u_q|j>``
          * ``q := Mi - Mj = (-1, 0, 1)``
          * ``u_q`` is the ``q``th component of the magnetic dipole operator in
            spherical coordinates.

        NB with this definition, the Rabi frequency is given by::

          * ``hbar * W = B_-q * R``
          * ``t_pi = pi/W``
          * where ``B_-q`` is the ``-q``th component of the magnetic field in spherical
            coordinates.
        """
        self.M1 = np.zeros((self.num_states, self.num_states))
        I = self.I
        I_dim = np.rint(2.0 * I + 1).astype(int)
        eyeI = np.identity(I_dim)

        for level, data in self.levels.items():
            lev = data.get_slice()
            J_dim = np.rint(2.0 * level.J + 1).astype(int)
            dim = J_dim * I_dim
            eyeJ = np.identity(J_dim)

            # magnetic dipole operator in spherical coordinates
            Jp = np.kron((-1 / np.sqrt(2)) * ap.operators.Jp(level.J), eyeI)
            Jm = np.kron((+1 / np.sqrt(2)) * ap.operators.Jm(level.J), eyeI)
            Jz = np.kron(ap.operators.Jz(level.J), eyeI)

            up = -data.g_J * _uB * Jp
            um = -data.g_J * _uB * Jm
            uz = -data.g_J * _uB * Jz

            if self.I > 0:
                Ip = np.kron(eyeJ, (-1 / np.sqrt(2)) * ap.operators.Jp(I))
                Im = np.kron(eyeJ, (+1 / np.sqrt(2)) * ap.operators.Jm(I))
                Iz = np.kron(eyeJ, ap.operators.Jz(I))

                up += data.g_I * _uN * Ip
                um += data.g_I * _uN * Im
                uz += data.g_I * _uN * Iz

            u = [um, uz, up]

            Mj = np.tile(self.M[lev], (dim, 1))
            Mi = Mj.T
            Q = Mi - Mj

            valid = np.abs(Q) <= 1
            valid[np.diag_indices(dim)] = False

            M1 = np.zeros((dim, dim))
            V = self.V[lev, lev]
            for transition in np.nditer(np.nonzero(valid)):
                i = transition[0]
                j = transition[1]
                q = np.rint(Q[i, j]).astype(int)

                psi_i = V[:, i]
                psi_j = V[:, j]

                M1[i, j] = ((-1) ** (q + 1)) * psi_i.conj().T @ u[q + 1] @ psi_j
            self.M1[lev, lev] = M1
