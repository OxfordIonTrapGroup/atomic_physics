import numpy as np

from atomic_physics.atoms.two_state import TwoStateAtom, field_for_frequency
from atomic_physics.bloch import Bloch
from atomic_physics.core import RFDrive
from atomic_physics.polarization import SIGMA_PLUS_POLARIZATION

omega = 1.2451158e6
delta = omega / 2

f0 = 100e6 * 2 * np.pi
b = field_for_frequency(f0)
atom = TwoStateAtom(b)

bloch = Bloch(atom)

H = bloch.H_for_rf_drive(
    level=TwoStateAtom.ground_level,
    drive=RFDrive(
        frequency=f0 - delta, amplitude=10e-6, polarization=SIGMA_PLUS_POLARIZATION
    ),
)

t = np.linspace(0, 10e-6, 1000)
results = bloch.solve_schroedinger(H, np.array([1, 0]), t)
P_julia = np.abs(results[:, 0]) ** 2

omega_eff = np.sqrt(omega**2 + delta**2)
P_analytic = 1 - (omega / omega_eff * np.sin(0.5 * omega_eff * t)) ** 2

np.testing.assert_allclose(P_julia, P_analytic, atol=5e-6, rtol=5e-6)
