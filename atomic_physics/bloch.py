import numpy as np
import scipy.constants as consts

from atomic_physics.core import Atom, Level, RFDrive
from atomic_physics.polarization import (
    PI_POLARIZATION,
    SIGMA_MINUS_POLARIZATION,
    SIGMA_PLUS_POLARIZATION,
)

try:
    import juliacall

    HAS_JULIA = True
except ModuleNotFoundError:
    HAS_JULIA = False
    # pytype: skip-file

if HAS_JULIA:
    jl = juliacall.newmodule("Bloch")
    jl.seval(
        """
        using QuantumOptics
        using PythonCall

        function make_hamiltonian(
            num_states::Int64,
            rabi_matrix::Matrix{ComplexF64},
            detuning_matrix::Matrix{Float64}
        )
            basis = NLevelBasis(num_states)

            elements = findall(x -> x != 0., rabi_matrix)
            hamiltonians = Vector{QuantumOpticsBase.TimeDependentSum}(undef, length(elements))

            for (hamiltonian_index, state_index) in enumerate(elements)
                rabi = rabi_matrix[state_index]
                detuning = detuning_matrix[state_index]

                sigma_plus = transition(basis, state_index[1], state_index[2])

                hamiltonian_static = 1 / 2 * rabi * sigma_plus
                hamiltonian_half = TimeDependentSum((t->exp(-1im*detuning*t))=>hamiltonian_static)
                hamiltonians[hamiltonian_index] = hamiltonian_half + dagger(hamiltonian_half)
            end
            return sum(hamiltonians)
        end
        """
    )


class Bloch:
    def __init__(self, atom: Atom):
        """Optical Bloch equations.

        Implemented as a thin wrapper around the ``QuantumOptics.jl`` toolkit using the
        ``JuliaCall`` library.

        Currently only RF (M1) transitions are supported; electric transitions and
        spontaneous emission are not (yet) supported.
        """
        if not HAS_JULIA:
            raise RuntimeError("JuliaCall not found. Try `poetry install -E bloch`.")

        self.atom: Atom = atom
        self.basis: juliacall.AnyValue = jl.NLevelBasis(self.atom.num_states)

    def hamiltonian_for_rf_drive(self, level: Level, drive: RFDrive):
        """Returns the (time-dependent) Hamiltonian describing the interaction between
        the atom and an RF drive.

        We make the approximation that the RF drive only couples to states within a
        single (specified) level.

        :param level: the level which the RF drive couples to.
        :param drive: the :class:`~atomic_physics.core.RFDrive`.
        :return: the Hamiltonian. This is a ``JuliaCall`` wrapper around a
            ``QuantumOptics.jl`` ``TimeDependentSum`` object.
        """
        level_slice = self.atom.get_slice_for_level(level)

        R = self.atom.get_magnetic_dipoles()[level_slice, level_slice]

        M = self.atom.M[level_slice]
        M_j, M_i = np.meshgrid(M, M)
        dMs = M_i - M_j  # dMs[i, j] = M[i] - M[j]

        rabi_matrix = np.zeros(
            (self.atom.num_states, self.atom.num_states), dtype=np.complex128
        )
        level_rabis = rabi_matrix[level_slice, level_slice]

        for dM, pol in [
            (+1.0, SIGMA_PLUS_POLARIZATION),
            (-1.0, SIGMA_MINUS_POLARIZATION),
            (0.0, PI_POLARIZATION),
        ]:
            amplitude = drive.amplitude * np.vdot(drive.polarization, pol)

            if amplitude == 0.0:
                continue

            # pol_inds[upper, lower] = True if (M[upper] - M[lower] = dM) else False
            # pol_inds[lower, upper] = False
            # pol_inds[idx, idx] = False
            pol_inds = np.triu(dMs == dM)
            level_rabis[pol_inds] = amplitude * R[pol_inds] / consts.hbar

        detuning_matrix = np.zeros(
            (self.atom.num_states, self.atom.num_states), dtype=np.float64
        )
        state_energies = self.atom.state_energies[level_slice]
        E_j, E_i = np.meshgrid(state_energies, state_energies)
        transition_frequency = E_i - E_j  # freq[upper, lower] = E[upper] - E[lower]
        detuning_matrix[level_slice, level_slice] = (
            drive.frequency - transition_frequency
        )

        return jl.make_hamiltonian(
            self.atom.num_states,
            jl.Matrix[jl.Complex[jl.Float64]](rabi_matrix),
            jl.Matrix[jl.Float64](detuning_matrix),
        )

    def make_ket(self, state_vector: np.ndarray):
        """Converts a numpy array representation of a state vector into a ``JuliaCall``
        wrapper around a ``QuantumOptics.jl`` ``Ket`` object.

        :param psi0: vector of complex coefficients describing the atom's initial state.
        :return: the corresponding ``Ket``.
        """
        if state_vector.shape != (self.atom.num_states,):
            raise ValueError(
                "Invalid state vector shape. "
                f"Got {state_vector.shape}, expected {(self.atom.num_states,)}"
            )
        return jl.Ket(
            self.basis,
            jl.convert(
                jl.Vector[jl.Complex[jl.Float64]],
                np.asarray(state_vector, dtype=np.complex128),
            ),
        )

    def solve_schroedinger(
        self, hamiltonian, psi0: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Integrate the time-dependent Schroedinger equation for a given Hamiltonian
        and initial state.

        :param hamiltonian: the Hamiltonian.
        :param psi0: vector of complex coefficients describing the atom's initial state.
        :param t: vector of time points to return the atom's state at.
        :return: ``(len(t), atom.num_states)`` array of complex coefficients describing
            the atom's state at each time ``t``. The first dimension of the returned
            array corresponds to points in the input vector ``t``, while the second
            dimension corresponds to the states in the atom.
        """
        _, psi_t = jl.timeevolution.schroedinger_dynamic(
            jl.convert(jl.Vector[jl.Float64], t),
            self.make_ket(psi0),
            hamiltonian,
        )

        return np.array([psi.data.to_numpy() for psi in psi_t])
