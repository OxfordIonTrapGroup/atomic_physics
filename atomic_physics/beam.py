import cmath
from dataclasses import dataclass, field
import numpy as np
import scipy.constants as consts

"""Vectors to convert a Cartesian vector to the spherical basis, with q=-1, 0, 1
    corresponding to the components of the beam that can cause absorbtion between
    states with M_u - M_l = q."""
C1_P1 = 1/np.sqrt(2) * np.array([-1., -1.j, 0])
C1_0 = np.array([0, 0, 1.])
C1_M1 = 1/np.sqrt(2) * np.array([1., -1.j, 0])

"""Matrices to convert a Cartesian tensor to the spherical basis, with q=-2, -1, 0, 1, 2
    corresponding to the components of the beam that can cause absorbtion between
    states with M_u - M_l = q."""
C2_P2 = 1/np.sqrt(6) * np.array([[1, 1.j, 0], [1.j, -1, 0], [0, 0, 0]])
C2_P1 = 1/np.sqrt(6) * np.array([[0, 0, -1], [0, 0, -1.j], [-1, -1.j, 0]])
C2_0 = 1/3 * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
C2_M1 = 1/np.sqrt(6) * np.array([[0, 0, 1], [0, 0, -1.j], [1, -1.j, 0]])
C2_M2 = 1/np.sqrt(6) * np.array([[1, -1.j, 0], [-1.j, -1, 0], [0, 0, 0]])

@dataclass
class Beam:
    """Contains all parameters and properties of a circular, Gaussian laser beam.

    The beam takes a total power and a (minimum) waist radius, defined as the 1/e^2
    intensity radius.

    The beam polarisation is defined using the three angles phi, gamma and theta.
    Phi is the angle between the beam and the B-field, such that if they are 
    parallel, phi=0. The beam direction and the B-field then define a plane: we
    decompose the polarisation vector into two unit vectors, one in this plane 
    (k1) and one perpendicular to it (k2). The beam polarisation is then defined
    as 
        e = cos(gamma) k1 + exp(i theta) sin(gamma) k2
    Gamma = 0 gives linear polarisation that has maximal projection along the B-field.
    Gamma = +- pi/4, theta = 0 gives linear polarisation along +-45 deg to the 
        plane.
    Gamma = pi/4, theta = +-pi/2 give circular polarisations.
    
    :param wavelength: The wavelength of the light in m.
    :param waist_radius: The (minimum) waist radius of the beam in m.
    :param power: The power in the beam in W.
    :param phi: The angle in rad between the beam and the B-field.
    :param gamma: The angle in rad between the (complex) polarisation vector
        and the plane described by the B-field and beam vector.
    :param theta: The phase angle in rad between the polarisation component in the
        plane and out of the plane."""
    wavelength: float
    waist_radius: float
    power: float
    phi: float
    gamma: float
    theta: float
    I_peak: float = field(init=False)
    E_field: float = field(init=False)  # The magnitude of the E-field at the beam centre
    rayleigh_range: float = field(init=False) 
    omega: float = field(init=False)

    def __post_init__(self):
        self.I_peak = 2 * self.power / (np.pi * self.waist_radius**2)
        self.E_field = np.sqrt(2 * self.I_peak / (consts.c * consts.epsilon_0))
        self.rayleigh_range = np.pi * self.waist_radius**2 / self.wavelength
        self.omega = consts.c / self.wavelength * 2 * np.pi

    def beam_direction_cartesian(self):
        """Return the unit vector along the beam direction in Cartesian 
            coordinates [x, y, z]"""
        return [np.sin(self.phi), 0, np.cos(self.phi)]

    def polarisation_cartesian(self):
        """Return a list of the Cartesian components of the (complex) polarisation
            vector [x, y, z]"""
        return [
            np.cos(self.phi) * np.cos(self.gamma),
            cmath.exp(1j * self.theta) * np.sin(self.gamma),
            -np.sin(self.phi) * np.cos(self.gamma),
            ]
    
    def polarisation_rank1_polar(self):
        """
        Returns the rank-1 tensor components of the polarisation vector that drive
        dipole transitions, i.e. the sigma_m, sigma_p and pi components of the 
        polarisation vector. Output is a list of the form [sigma_m, pi, sigma_p]
        in polar form.
        """
        return [cmath.polar(q) for q in self.polarisation_rank1()]
    
    def polarisation_rank1(self):
        """
        Returns the rank-1 tensor components of the polarisation vector that drive
        dipole transitions, i.e. the sigma_m, sigma_p and pi components of the 
        polarisation vector. Output is a list of the form [sigma_m, pi, sigma_p]
        in rectangular form (i.e. re + j im).
        """
        vect = np.array(self.polarisation_cartesian())
        sigma_m = np.sum(vect * C1_M1)
        pi = np.sum(vect * C1_0)
        sigma_p = np.sum(vect * C1_P1)
        return [sigma_m, pi, sigma_p]
    
    def polarisation_rank2_polar(self):
        """
        Returns the rank-2 tensor components of the polarisation tensor, which 
        drive quadrupole transitions. Output is a list containing the five components
        in increasing order from q_m2 to q_p2 in polar form.
        """
        return [cmath.polar(q) for q in self.polarisation_rank2()]
    
    def polarisation_rank2(self):
        """
        Returns the rank-2 tensor components of the polarisation tensor, which 
        drive quadrupole transitions. Output is a list containing the five components
        in increasing order from q_m2 to q_p2 in rectangular form (i.e. re + j im).
        """
        beam_vect = np.array(self.beam_direction_cartesian())
        pol_vect = np.array(self.polarisation_cartesian())
        
        q_m2 = pol_vect.T @ C2_M2 @ beam_vect
        q_m1 = pol_vect.T @ C2_M1 @ beam_vect
        q_0 = pol_vect.T @ C2_0 @ beam_vect
        q_p1 = pol_vect.T @ C2_P1 @ beam_vect
        q_p2 = pol_vect.T @ C2_P2 @ beam_vect

        return [q_m2, q_m1, q_0, q_p1, q_p2]

    def print_rank1(self, title=None):
        """Pretty-print the rank-1 tensor components of the beam."""
        if title is None:
            title = ""
        sigma_m, pi, sigma_p = self.polarisation_rank1_rect()
        print(title)
        print(f"sigma_m: {sigma_m:.3f}")
        print(f"pi: {pi:.3f}")
        print(f"sigma_p: {sigma_p:.3f}")


    def print_rank2(self, title=None):
        """Pretty-print the rank-2 tensor components of the beam."""
        if title is None:
            title = ""
        print(title)
        for idx, p in enumerate(self.polarisation_rank1_rect()):
            print(f"q_{idx-2}: {p:.3f}")