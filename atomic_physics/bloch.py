import juliacall
import numpy as np
import scipy.constants as consts

from atomic_physics.core import Atom, Level, RFDrive
from atomic_physics.polarization import (
    PI_POLARIZATION,
    SIGMA_MINUS_POLARIZATION,
    SIGMA_PLUS_POLARIZATION,
)

jl = juliacall.newmodule("Bloch")
jl.seval(
    """
    using QuantumOptics
    using PythonCall

    function make_rf_hamiltonian(
        num_states::Int32,
        Omegas::PythonCall.Wrap.PyArray,
        deltas::PythonCall.Wrap.PyArray
    )
        basis = NLevelBasis(num_states)

        Hs = QuantumOpticsBase.TimeDependentSum[]
        for element in eachindex(Omegas)
            if Omegas[element] == 0.
                continue
            end

            Omega = Omegas[element]
            delta = deltas[element]

            op = 1 / 2 * Omega * transition(basis, element[1], element[2])
            H_half = TimeDependentSum((t->exp(-1im*delta*t))=>op)
            H = H_half + dagger(H_half)
            push!(Hs, H)
        end
        return sum(Hs)
    end

    function make_ket(
        num_states::Int32,
        state_vector::PythonCall.Wrap.PyArray
    )
        basis = NLevelBasis(num_states)
        return Ket(basis, Vector{ComplexF64}(state_vector))
    end
    """
)


class Bloch:
    def __init__(self, atom: Atom):
        """Optical Bloch equations.

        This is implemented as a thin wrapper around the ``QuantumOptics.jl`` toolkit
        using the ``JuliaCall`` library.

        Currently, only RF (M1) transitions are supported; electric transitions and
        spontaneous emission are not (yet) supported.
        """
        self.atom: Atom = atom
        self.basis: juliacall.AnyValue = jl.NLevelBasis(np.int32(self.atom.num_states))

    def H_for_rf_drive(self, level: Level, drive: RFDrive) -> juliacall.AnyValue:
        """Returns the (time-dependent) Hamiltonian describing the interaction between
        the atom and an RF drive.

        We make the approximation that the RF drive only couples to states within a
        single level.

        :param level: the level which the RF drive couples to.
        :param drive: the :class:`~atomic_physics.core.RFDrive`.
        :return: the Hamiltonian. This is a JuliaCall wrapper around a ``QuantumOptics.jl``
            `TimeDependentSum` object.
        """
        R = self.atom.get_magnetic_dipoles()
        M_j, M_i = np.meshgrid(self.atom.M, self.atom.M)
        dMs = M_i - M_j

        Omegas = np.zeros(R.shape, dtype=np.complex64)
        for dM, pol in [
            (+1, SIGMA_PLUS_POLARIZATION),
            (-1, SIGMA_MINUS_POLARIZATION),
            (0, PI_POLARIZATION),
        ]:
            amplitude = drive.amplitude * np.vdot(drive.polarization, pol)

            if amplitude == 0.0:
                continue

            pol_inds = np.triu(dMs == dM)
            Omegas[pol_inds] = amplitude * R[pol_inds] / consts.hbar

        E_j, E_i = np.meshgrid(self.atom.state_energies, self.atom.state_energies)
        freq = E_i - E_j
        deltas = freq - drive.frequency

        return jl.make_rf_hamiltonian(
            np.int32(self.atom.num_states),
            np.asarray(Omegas, dtype=np.complex128),
            np.asarray(deltas, dtype=np.float64),
        )

    def make_ket(self, state_vector: np.array) -> juliacall.AnyValue:
        """Converts a numpy array representation of a state vector into a ``JuliaCall``
        wrapper around a ``QuantumOptics.jl`` ``Ket`` object.

        :param state_vector: array of complex amplitudes describing the atom's state.
        :return: the corresponding ``Ket``.
        """
        if state_vector.shape != (self.atom.num_states,):
            raise ValueError(
                "Invalid state vector shape. "
                f"Got {state_vector.shape}, expected {(1, self.atom.num_states)}"
            )
        return jl.make_ket(
            np.int32(self.atom.num_states),
            np.asarray(state_vector, dtype=np.complex128),
        )

    def solve_schroedinger(
        self, H: juliacall.AnyValue, psi0: np.array, t: np.array
    ) -> np.array:
        """Integrate the time-dependent Schroedinger equation for a given Hamiltonian
        and initial state.

        :param H: the Hamiltonian.
        :param psi0: vector of complex coefficients describing the atom's initial state.
        :param t: vector of time points to return the atom's state at.
        :return: ``(len(t), atom.num_states)`` array of complex coefficients describing
            the atom's state at each time ``t``. The first dimension of the returned
            array corresponds to points in the input vector ``t``, while the second
            dimension corresponds to the states in the atom.
        """
        _, psi_t = jl.timeevolution.schroedinger_dynamic(
            np.asarray(t, dtype=np.float64),
            self.make_ket(psi0),
            H,
        )

        return np.array([psi.data.to_numpy() for psi in psi_t])
