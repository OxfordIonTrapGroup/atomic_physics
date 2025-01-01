import dataclasses

import numpy as np
import scipy.constants as consts

from atomic_physics import operators
from atomic_physics.wigner import wigner3j

_uB = consts.physical_constants["Bohr magneton"][0]
_uN = consts.physical_constants["nuclear magneton"][0]


@dataclasses.dataclass(frozen=True)
class Level:
    """Represents a single level.

    Attributes:
        n: the level's principal quantum number
        L: the level's orbital angular momentum
        J: the level's total (spin + orbital) electronic angular momentum
        S: the level's spin angular momentum
    """

    n: int
    L: float
    J: float
    S: float


@dataclasses.dataclass
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
            g_L = 1
            g_S = -consts.physical_constants["electron g factor"][0]

            S = self.level.S
            J = self.level.J
            L = self.level.L

            self.g_J = g_L * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (
                2 * J * (J + 1)
            ) + g_S * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))


@dataclasses.dataclass(frozen=True)
class LevelStates:
    r"""Stores information about the states within a level.

    Attributes:
        freq: frequency (rad/s) of the centre-of-gravity transition from the ground-level
            to this level. These frequencies are calculated automatically by combining
            data from atom's transitions. In cases where the the level structure is multiply connected,
            taking different routes between the same levels may give slightly different
            values for the level frequencies due to inconsistencies between the measured
            transition frequencies. We do not make guarantees about which route is taken
            for these calculations and this data should not be relied on for accurate
            calculations (see :class:`Atom`\'s ``get_transition_frequency_for_states`` method
            for that).
        start_index: index into the state vector for the first (highest energy) state within
            this level.
        stop_index: index into the state vector of the laset (lowest energy) state within
            this level.
        num_states: the number of states within the level.
    """

    frequency: float
    start_index: int
    stop_index: int
    num_states: int


@dataclasses.dataclass(frozen=True)
class Transition:
    r"""Represents a transition between a pair of :class:`~.Level`\s.

    Attributes:
        lower: the :class:`Level` in the transition with lower energy
        upper:  the :class:`Level` in the transition with greater energy
        freq: the transition frequency (rad/s)
        einstein_A: the transition's Einstein A coefficient
    """

    lower: Level
    upper: Level
    frequency: float
    einstein_A: float


@dataclasses.dataclass
class Laser:
    """Represents a laser.

    Attributes:
        transition: string giving the name of the transition driven by the laser.
        polarization: the laser's polarization, defined as the difference in angular
            momentum between the higher and lower energy states coupled by this laser
            (``q = M_upper - M_lower``). ``q = +1`` for σ+ light, ``q = -1`` for σ-
            light and ``q = 0`` for π light.
        intensity: the laser intensity (saturation intensities).
        detuning: the laser's detuning (rad/s) from the transition centre of gravity,
           defined so that ``w_laser = w_transition + detuning``.
    """

    transition: str
    polarization: int
    intensity: float
    detuning: float


@dataclasses.dataclass(frozen=True)
class RFDrive:
    """Represents an AC magnetic field, which drives magnetic dipole transitions.

    Attributes:
        frequency: frequency of the RF drive (rad/s).
        amplitude: amplitude of the field (T).
        polarization: Jones vector describing the magnetic field's polarization.
    """

    frequency: float
    amplitude: float
    polarization: np.ndarray

    def __post_init__(self):
        if self.polarization.shape != (3,):
            raise ValueError("Polarization must be a 3-element vector")


@dataclasses.dataclass
class Atom:
    r"""Represents a specific atom at a given magnetic field.

    Attributes:
        magnetic_field: the applied magnetic field (Tesla)
        level_data: atomic structure dictionary, mapping :class:`~.Level`\s to
            :class:`~.LevelData`.
        level_states: level states dictionary, mapping :class:`~.Level`\s to
            :class:`~.LevelStates`.
        transitions: transitions dictionary, mapping names of transitions to
            :class:`~.Transition` objects.
        nuclear_spin: Nuclear spin.

        num_states: the total number of states contained within the atom.
        state_energies: vector of energies of each state in the atom. States are ordered
            by *decreasing* energy, with ``state_energies[0]`` corresponding the state with
            highest energy and ``state_energies[-1]`` corresponding to the ground state.
        state_vectors: array of state eigenvectors in the high-field basis. The vector
            ``state_vectors[:, state_index]`` gives the representation of the state
            with energy ``state_energies[state_index]`` in the basis of high-field
            energy eigenstates (``M_I``, ``M_J``). NB we use the high-field basis
            internally because it is the most convenient to represent our operators in.
            These states are not energy-ordered and are not generally energy eigenstates.
        high_field_M_I: array of values of ``M_I`` for each state in the high-field
            basis used in ``state_vectors``.
        high_field_M_J: array of values of ``M_J`` for each state in the high-field
            basis used in ``state_vectors``.
        M: vector of magnetic quantum numbers for each state.
        F: vector of total (electron + nucleus) angular momentum for each state. This is
            not generally a good quantum number, however provide an estimate using
            ``<F^2> = f * (f + 1)`` and rounding to the closest valid value. This is
            useful when the field is sufficiently low.
        M_I: vector of nuclear magnetic quantum numbers for each state. This is not
            generally a good quantum number, however provide an estimate using
            ``<Iz> = M_I`` and rounding to the closest valid value. This is useful when
            the field is sufficiently high.
        M_J: vector of electron magnetic quantum numbers for each state. This is not
            generally a good quantum number, however provide an estimate using
            ``<Jz> = M_J`` and rounding to the closest valid value. This is useful when
            the field is sufficiently high.
        _electric_multipoles: electric multiple matrix (scattering amplitudes between
            states). See :meth:`get_electric_multipoles`. To keep construction fast,
            we calculate this lazily when needed.
        _magnetic_dipoles: magnetic dipole matrix elements. To keep construction fast,
            we calculate this lazily when needed.
    """

    magnetic_field: float
    level_data: dict[Level, LevelData]
    level_states: dict[Level, LevelStates]
    transitions: dict[str, Transition]
    nuclear_spin: float

    num_states: int = dataclasses.field(init=False)
    state_energies: np.ndarray = dataclasses.field(init=False)
    state_vectors: np.ndarray = dataclasses.field(init=False)
    high_field_M_I: np.ndarray = dataclasses.field(init=False)
    high_field_M_J: np.ndarray = dataclasses.field(init=False)
    M: np.ndarray = dataclasses.field(init=False)

    F: np.ndarray = dataclasses.field(init=False)
    M_I: np.ndarray = dataclasses.field(init=False)
    M_J: np.ndarray = dataclasses.field(init=False)

    _electric_multipoles: np.ndarray | None = dataclasses.field(
        init=False, default=None
    )
    _magnetic_dipoles: np.ndarray | None = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        self.num_states = max(
            [states.stop_index for states in self.level_states.values()]
        )

        self.state_energies = np.zeros(self.num_states)
        self.state_vectors = np.zeros((self.num_states, self.num_states))
        self.high_field_M_I = np.zeros(self.num_states)
        self.high_field_M_J = np.zeros(self.num_states)
        self.M = np.zeros(self.num_states)

        self.F = np.zeros(self.num_states)
        self.M_I = np.zeros(self.num_states)
        self.M_J = np.zeros(self.num_states)

        I_dim = np.rint(2.0 * self.nuclear_spin + 1).astype(int)

        for level, level_data in self.level_data.items():
            J_dim = np.rint(2.0 * level.J + 1).astype(int)

            Jp = np.kron(operators.Jp(level.J), np.identity(I_dim))
            Jm = np.kron(operators.Jm(level.J), np.identity(I_dim))
            Jz = np.kron(operators.Jz(level.J), np.identity(I_dim))

            Ip = np.kron(np.identity(J_dim), operators.Jp(self.nuclear_spin))
            Im = np.kron(np.identity(J_dim), operators.Jm(self.nuclear_spin))
            Iz = np.kron(np.identity(J_dim), operators.Jz(self.nuclear_spin))

            H = level_data.g_J * _uB * self.magnetic_field * Jz

            if self.nuclear_spin != 0:
                gI = level_data.g_I
                IdotJ = Iz @ Jz + (1 / 2) * (Ip @ Jm + Im @ Jp)

                H += -gI * _uN * self.magnetic_field * Iz
                H += level_data.Ahfs * IdotJ

                if level.J > 1 / 2 and self.nuclear_spin > 1 / 2:
                    IdotJ2 = np.linalg.matrix_power(IdotJ, 2)
                    ident = np.identity(I_dim * J_dim)
                    H += (
                        level_data.Bhfs
                        / (
                            2
                            * self.nuclear_spin
                            * level.J
                            * (2 * self.nuclear_spin - 1)
                            * (2 * level.J - 1)
                        )
                        * (
                            3 * IdotJ2
                            + (3 / 2) * IdotJ
                            - ident
                            * self.nuclear_spin
                            * (self.nuclear_spin + 1)
                            * level.J
                            * (level.J + 1)
                        )
                    )

            H /= consts.hbar  # work in angular frequency units

            state_energies, state_vectors = np.linalg.eig(H)

            # sort in terms of *decreasing* energy
            inds = np.argsort(state_energies)[::-1]
            state_vectors = state_vectors[:, inds]
            state_energies = state_energies[inds]

            # check that the eigensolver found the angular momentum eigenstates
            M = np.diag(state_vectors.conj().T @ (Iz + Jz) @ state_vectors)
            if max(abs(M - np.rint(2 * M) / 2)) > 1e-5:
                raise ValueError(
                    "Error finding angular momentum eigenstates. Is the field too "
                    "small to lift the state degeneracy?"
                )

            M = np.rint(2 * M) / 2

            level_slice = self.get_slice_for_level(level)

            self.state_energies[level_slice] = state_energies
            self.state_vectors[level_slice, level_slice] = state_vectors
            self.high_field_M_I[level_slice] = np.kron(
                np.ones(J_dim), np.arange(-self.nuclear_spin, self.nuclear_spin + 1)
            )
            self.high_field_M_J[level_slice] = np.kron(
                np.arange(-level.J, level.J + 1), np.ones(I_dim)
            )

            self.M[level_slice] = M

            # F, M_I & M_J aren't generally good quantum numbers, but find the closest
            # value anyway since it's useful in cases where we're "close enough" to
            # high or low field.
            Fz = Iz + Jz
            Fp = Ip + Jp
            Fm = Im + Jm

            F_2_op = Fz @ Fz + (1 / 2) * (Fp @ Fm + Fm @ Fp)
            F_2 = np.diag(state_vectors.conj().T @ F_2_op @ state_vectors)  # <F^2>

            F = 0.5 * (np.sqrt(1 + 4 * F_2) - 1)  # <F^2> = f * (f + 1)
            M_I = np.diag(state_vectors.conj().T @ (Iz) @ state_vectors)  # M_I = <Iz>
            M_J = np.diag(state_vectors.conj().T @ (Jz) @ state_vectors)  # M_J = <Jz>

            def closest(number, valid_values):
                return valid_values[np.abs(number - valid_values).argmin()]

            valid_M_I = np.arange(-self.nuclear_spin, self.nuclear_spin + 1)
            valid_M_J = np.arange(-level.J, level.J + 1)

            F_list = np.arange(
                abs(self.nuclear_spin - level.J), self.nuclear_spin + level.J + 1
            )
            if level_data.Ahfs > 0:
                F_list = F_list[::-1]

            for M in set(self.M[level_slice]):
                for Fidx, idx in np.ndenumerate(np.where(M == self.M[level_slice])):
                    self.F[level_slice][idx] = F_list[abs(M) <= F_list][Fidx[1]]

            self.M_I[level_slice] = list(map(lambda x: closest(x, valid_M_I), M_I))
            self.M_J[level_slice] = list(map(lambda x: closest(x, valid_M_J), M_J))

    def get_transition_frequency_for_states(
        self, states: tuple[int, int], relative: bool = True
    ) -> float:
        """Returns the frequency (angular units) of the transition between a pair of
        states.

        :param states: tuple of indices of the states involved in the transition.
        :param relative: if ``False`` the returned frequency is the absolute frequency
            of the transition between the two states. If ``True`` we subtract the
            centre of gravity frequency of the transition between the two levels the
            states lie in to calculate the frequency relative to the overall
            transition's centre of gravity.
        :return: the transition frequency (rad/s).
        """
        if len(states) != 2:
            raise ValueError(f"Expected 2 state indices, got {len(states)}.")

        lower = max(states)
        upper = min(states)

        f_rel = self.state_energies[upper] - self.state_energies[lower]

        if relative:
            return f_rel

        transition = self.get_transition_for_levels(
            levels=(self.get_level_for_state(lower), self.get_level_for_state(upper))
        )
        f_abs = f_rel + transition.frequency

        return f_abs

    def get_slice_for_level(self, level: Level) -> slice:
        """Returns a slice object, which can be used to index into a state vector to
        select only the states which lie within a given level.

        :param level: the level to select.
        :return: the slice object.
        """
        states = self.level_states[level]
        return slice(states.start_index, states.stop_index)

    def get_states_for_M(self, level: Level, M: float) -> np.ndarray:
        """Returns the indicates of the states within a given level with the specified
        value of :math:`M`.

        :param level: the level to look within.
        :param M: the value of :math:`M` to look for.
        :return: array of state indices.
        """
        inds = np.arange(self.num_states)
        level_states = self.level_states[level]

        level_states = np.logical_and(
            level_states.start_index <= inds, inds < level_states.stop_index
        )

        M_states = np.logical_and(
            level_states,
            self.M[inds] == M,
        )

        return np.array(inds[M_states])

    def get_state_for_F(self, level: Level, F: float, M_F: float) -> int:
        """Returns the index of the state with a given value of :math:`F` and
        :math:`M_F` within a given level.

        :math:`F` is only a good quantum number at zero field. At other fields, this is a
        "best guess" at the right state (see :attr:`.F`).

        :param level: the level to look within.
        :param F: the value of :math:`F` to look for.
        :param M_F: the value of :math:`M_F` to look for.
        :return: the index of the corresponding state.
        """
        M_inds = self.get_states_for_M(level, M=M_F)
        F_inds = M_inds[self.F[M_inds] == F]

        if len(F_inds) != 1:
            raise ValueError(f"No unique state with F={F} found in level {level}")

        return F_inds.ravel()[0]

    def get_state_for_MI_MJ(self, level: Level, M_I: float, M_J: float) -> int:
        """Returns the index of the state with a given value of :math:`M_I` and
        :math:`M_J` within a given level.

        :math:`M_I` and :math:`M_J` are only good quantum numbers at high field. At
        other fields, this is a "best guess" at the right state (see :attr:`.M_I` and
        :attr:`M_J`).

        :param level: the level to look within.
        :param M_I: the value of :math:`M_I` to look for.
        :param M_J: the value of :math:`M_J` to look for.
        :return: the index of the corresponding state.
        """
        inds = np.arange(self.num_states)
        level_states = self.level_states[level]

        level_states = np.logical_and(
            level_states.start_index <= inds, inds < level_states.stop_index
        )

        M_states = np.logical_and(self.M_I[inds] == M_I, self.M_J[inds] == M_J)

        states = np.logical_and(level_states, M_states)

        if len(states) != 1:
            raise ValueError(
                f"No unique state with M_I={M_J}, M_J={M_J} found in level {level}"
            )

        return states.ravel()[0]

    def get_level_for_state(self, state: int) -> Level:
        """Returns the level a state lies in.

        :param state: index of the state.
        :return: the :class:`Level` the state lies within.
        """
        for level, level_states in self.level_states.items():
            if state >= level_states.start_index and state < level_states.stop_index:
                return level

        raise ValueError(f"No state with index {state}")

    def get_transition_for_levels(self, levels: tuple[Level, Level]) -> Transition:
        r"""Returns the transition between a pair of :class:`Level`\s.

        :param levels: tuple containing the two :class:`Level`\s involved in the
            transition.
        """
        if len(levels) != 2:
            raise ValueError(f"Expected 2 levels, got {len(levels)}.")

        for transition in self.transitions.values():
            if transition.lower in levels and transition.upper in levels:
                return transition

        raise ValueError(
            f"No transition found between levels {levels[0]} and {levels[1]}"
        )

    def get_population(
        self, state_vector: np.ndarray, inds: Level | int | slice
    ) -> float:
        """Returns the total population in a set of states.

        :param state: state vector
        :param states: set of states to sum over. This can be any of: a :class:`.Level`;
          a state index; or, a ``slice``.
        """
        if isinstance(inds, Level):
            return np.sum(state_vector[self.get_slice_for_level(inds)])
        elif not isinstance(inds, int) and not isinstance(inds, slice):
            raise TypeError("inds must be a level, slice or index")
        return np.sum(state_vector[inds])

    def get_saturation_intensity(self, transition: str) -> float:
        """Returns the saturation intensity for a transition.

        We adopt a convention whereby, for a resonantly-driven cycling
        transition, one saturation intensity gives equal stimulated and
        spontaneous transition rates.

        :param transition: the transition name
        :return: saturation intensity (W/m^2)
        """
        transition = self.transitions[transition]
        scattering_rates = np.power(np.abs(self._electric_multipoles), 2)
        total_scattering_rates = np.sum(scattering_rates, 0)
        stretched_state_index = self.level_states[transition.upper].start_index
        scattering_rate = total_scattering_rates[stretched_state_index]

        omega = transition.frequency
        return (
            consts.hbar * (omega**3) * scattering_rate / (6 * np.pi * (consts.c**2))
        )

    def intensity_to_power(
        self, transition: str, waist_radius: float, intensity: float
    ) -> float:
        """Returns the power needed to achieve a given intensity.

        :param transition: name of the transition to drive
        :param waist_radius: Gaussian beam waist (:math:`1/e^2` intensity radius in `m`)
        :param intensity: beam intensity (saturation intensities).
        :return: beam power (W)
        """
        sat = self.get_saturation_intensity(transition)
        return 0.5 * np.pi * (waist_radius**2) * intensity * sat

    def get_electric_multipoles(self) -> np.ndarray:
        """Returns the electric multi-pole matrix for each transition.

        The values we return are scattering amplitudes between states (square
        root of the spontaneous scattering rates), rather than matrix elements
        themselves, since they are somewhat more convenient to work with.

        Currently we only calculate dipole (E1) and Quadrupole (E2) scattering
        amplitudes.

        To do: define what we mean by a matrix element, and how the amplitudes
        stored in ePole are related to the matrix elements and to the Rabi
        frequencies.

        To do: double check the signs of the amplitudes.
        """
        if self._electric_multipoles is None:
            self._calc_electric_multipoles()
        return self._electric_multipoles

    def get_magnetic_dipoles(self) -> np.ndarray:
        r"""Returns an array of magnetic dipole matrix elements.

        The matrix elements are given by
        :math:`R_{ul} := (-1)^{q+1} \left<u|\mu_q|l\right>` where:


        * :math:`\left|u\right>` (:math:`\left|l\right>`) is the state with greater
          (lower) energy
        * :math:`q := Mu - Ml`.
        * :math:`\mu_q` is the :math:`q` th component of the magnetic dipole operator in
          the spherical basis.
        """
        if self._magnetic_dipoles is None:
            self._calc_magnetic_dipoles()
        return self._magnetic_dipoles

    def get_rabi_m1(self, lower: int, upper: int, amplitude: float) -> float:
        r"""Returns the Rabi frequency for a magnetic dipole transition.

        See also :meth:`get_magnetic_dipoles`.

        :param lower: index of the state with lower energy involved in the transition.
        :param upper: index of the state with higher energy involved in the transition.
        :param amplitude: amplitude of the component of the driving magnetic field (
            spherical basis) which couples to this transition (Tesla).
        :return: the Rabi frequency. We retain phase information, so this can be either
            positive or negative. We define the Rabi frequency so that
            :math:`t_{\pi} = \pi / |\Omega|`
        """
        R = self.get_magnetic_dipoles()[upper, lower]
        Omega = amplitude * R / consts.hbar
        return Omega

    def _calc_electric_multipoles(self):
        if self._electric_multipoles is not None:
            return

        # We first calculate the scattering amplitudes in the high-field basis where
        # we can forget about nuclear spin.
        ePole_hf = np.zeros((self.num_states, self.num_states))
        I_dim = np.rint(2.0 * self.nuclear_spin + 1).astype(int)

        for transition in self.transitions.values():
            dJ = transition.upper.J - transition.lower.J
            dL = transition.upper.L - transition.lower.L

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

            # We want to get the sub-matrix which has just the states involved in
            # this transition. Because the states from the two levels will generally
            # not be adjacent in our state vector we can't just use standard slice
            # notation. Instead, what we want is numy's "cross-product" indexing...
            #
            # Start by concatenating the slices to generate a single vector which
            # contains the indices of all states involved in this transition
            transition_states = np.r_[
                self.get_slice_for_level(transition.lower),
                self.get_slice_for_level(transition.upper),
            ]
            # numpy indexing magic: create an object we can use to select all elements
            # of an array which correspond to transitions between states involved in
            # this transition
            subspace = np.ix_(transition_states, transition_states)

            MJ_u = np.arange(-transition.upper.J, transition.upper.J + 1)
            MJ_l = np.arange(-transition.lower.J, transition.lower.J + 1)
            Jdim_u = int(np.rint(2 * transition.upper.J + 1))
            Jdim_l = int(np.rint(2 * transition.lower.J + 1))
            Jdim = Jdim_u + Jdim_l

            # High-field scattering rates for this transition (ignoring I)
            ePole_hf_subspace = np.zeros((Jdim, Jdim))
            for ind_u in range(Jdim_u):
                # q := MJ_u - MJ_l
                for q in range(-order, order + 1):
                    if abs(MJ_u[ind_u] - q) > transition.lower.J:
                        continue

                    ind_l = np.argwhere(q == MJ_u[ind_u] - MJ_l)
                    sign = (-1) ** (
                        2 * transition.upper.J
                        + transition.lower.J
                        - MJ_u[ind_u]
                        + order
                    )
                    ePole_hf_subspace[ind_l, ind_u + Jdim_l] = (
                        wigner3j(
                            transition.upper.J,
                            order,
                            transition.lower.J,
                            -MJ_u[ind_u],
                            q,
                            (MJ_u[ind_u] - q),
                        )
                        * sign
                    )
            ePole_hf_subspace *= np.sqrt(
                transition.einstein_A * (2 * transition.upper.J + 1)
            )

            # introduce the (still decoupled) nuclear spin
            ePole_hf[subspace] = np.kron(ePole_hf_subspace, np.identity(I_dim))

        # now couple...
        self._electric_multipoles = self.state_vectors.T @ ePole_hf @ self.state_vectors

    def _calc_magnetic_dipoles(self):
        if self._magnetic_dipoles is not None:
            return

        self._magnetic_dipoles = np.zeros((self.num_states, self.num_states))

        I_dim = np.rint(2.0 * self.nuclear_spin + 1).astype(int)
        eye_I = np.identity(I_dim)

        for level, data in self.level_data.items():
            J_dim = np.rint(2.0 * level.J + 1).astype(int)
            eye_J = np.identity(J_dim)

            dim = J_dim * I_dim

            # magnetic dipole operator in spherical coordinates
            Jp = np.kron((-1 / np.sqrt(2)) * operators.Jp(level.J), eye_I)
            Jm = np.kron((+1 / np.sqrt(2)) * operators.Jm(level.J), eye_I)
            Jz = np.kron(operators.Jz(level.J), eye_I)

            up = -data.g_J * _uB * Jp
            um = -data.g_J * _uB * Jm
            uz = -data.g_J * _uB * Jz

            if self.nuclear_spin > 0:
                Ip = np.kron(eye_J, (-1 / np.sqrt(2)) * operators.Jp(self.nuclear_spin))
                Im = np.kron(eye_J, (+1 / np.sqrt(2)) * operators.Jm(self.nuclear_spin))
                Iz = np.kron(eye_J, operators.Jz(self.nuclear_spin))

                up += data.g_I * _uN * Ip
                um += data.g_I * _uN * Im
                uz += data.g_I * _uN * Iz

            # R[i, j] := (-1)**(q+1)<i|u_q|j>``
            # q := M_upper - M_lower = (-1, 0, 1)
            # where M_upper (M_lower) is the magnetic quantum number for the state with
            # greater (lower) energy
            u = um - uz + up
            level_slice = self.get_slice_for_level(level)
            level_state_vectors = self.state_vectors[level_slice, level_slice]

            level_M = self.M[level_slice]
            Mj = np.tile(level_M, (dim, 1))
            Mi = Mj.T

            # Most of the possible transitions aren't allowed (zero matrix element)
            # so don't waste time on them.
            dM = np.rint(Mi - Mj).astype(int)
            valid = np.abs(dM) <= 1
            valid[np.diag_indices(dim)] = False

            level_dipoles = np.zeros((dim, dim))
            for i_ind, j_ind in np.nditer(np.nonzero(valid)):
                psi_i = level_state_vectors[:, i_ind]
                psi_j = level_state_vectors[:, j_ind]
                level_dipoles[i_ind, j_ind] = psi_i.conj().T @ u @ psi_j

            self._magnetic_dipoles[level_slice, level_slice] = level_dipoles


@dataclasses.dataclass
class AtomFactory:
    r"""Flexible interface for creating :class:`Atom`\s.

        Example usage:

        .. testcode::

            from atomic_physics.core import Atom
            from atomic_physics.ions.ca40 import Ca40


            atom: Atom = Ca40(magnetic_field=10e-4)

    Attributes:
        level_data: tuple of atomic structure data for each level in the atom
        transitions: tuple of transitions between levels
        nuclear_spin: the atom's nuclear spin
    """

    level_data: tuple[LevelData, ...]
    transitions: dict[str, Transition]
    nuclear_spin: float
    level_states: dict[Level, LevelStates] = dataclasses.field(init=False)

    def __post_init__(self):
        # Use transition data to find the absolute energy of each level and sort them
        # from highest energy to lowest
        processed_levels: dict[Level, float] = {}  # Level: energy (freq units)
        unprocessed_transition_names: list[str] = []

        if len(self.level_data) == 1:
            processed_levels[next(iter(self.level_data)).level] = 0.0
        else:
            unprocessed_transition_names += list(self.transitions.keys())
            transition = self.transitions[unprocessed_transition_names.pop()]
            processed_levels[transition.lower] = 0
            processed_levels[transition.upper] = transition.frequency

        while unprocessed_transition_names:
            for transition_name in unprocessed_transition_names:
                transition = self.transitions[transition_name]
                if transition.lower in processed_levels:
                    frequency = (
                        processed_levels[transition.lower] + transition.frequency
                    )
                    processed_levels[transition.upper] = frequency
                    break
                elif transition.upper in processed_levels:
                    frequency = (
                        processed_levels[transition.upper] - transition.frequency
                    )
                    processed_levels[transition.lower] = frequency
                    break
            else:
                raise ValueError(
                    "Transition '{}' would lead to a disconnected level"
                    " structure.".format(transition_name)
                )
            unprocessed_transition_names.remove(transition_name)

        levels = [level_data.level for level_data in self.level_data]
        if set(processed_levels.keys()) != set(levels):
            raise ValueError("Disconnected level structure")

        sorted_levels: list[tuple[Level, float]] = sorted(
            processed_levels.items(), key=lambda x: x[1], reverse=True
        )

        f_0 = sorted_levels[-1][1]  # ground-level energy
        start_index = 0
        level_states: dict[Level, LevelStates] = {}
        for level, level_freq in sorted_levels:
            num_states = int(np.rint((2 * self.nuclear_spin + 1) * (2 * level.J + 1)))
            level_states[level] = LevelStates(
                frequency=level_freq - f_0,
                start_index=start_index,
                stop_index=start_index + num_states,
                num_states=num_states,
            )
            start_index += num_states

        super().__setattr__("level_states", level_states)

    def __call__(self, magnetic_field: float) -> Atom:
        """Constructs an :class:`Atom`.

        :param magnetic_field: applied magnetic field (Tesla).
        """
        atom = Atom(
            magnetic_field=magnetic_field,
            level_data={level_data.level: level_data for level_data in self.level_data},
            level_states=self.level_states,
            transitions=self.transitions,
            nuclear_spin=self.nuclear_spin,
        )

        return atom

    def filter_levels(self, level_filter: tuple[Level, ...]) -> "AtomFactory":
        r"""Returns a new :class:`~.AtomFactory` which contains only a subset of the
        atomic levels.

        This method is used to avoid unnecessary calculations when not all atomic levels
        are needed.

        :param level_filter: list of :class:`Level`\s to include.
        :return: a new :class:`~.AtomFactory` containing only the selected subset of
            this :class:`~.AtomFactory`\'s levels.
        """
        level_data: tuple[LevelData, ...] = tuple(
            [
                level_data
                for level_data in self.level_data
                if level_data.level in level_filter
            ]
        )

        transitions = {
            transition_name: transition
            for transition_name, transition in self.transitions.items()
            if transition.lower in level_filter and transition.upper in level_filter
        }

        return AtomFactory(
            level_data=level_data,
            transitions=transitions,
            nuclear_spin=self.nuclear_spin,
        )
