.. _definitions:

Definitions and Conventions
###########################

Variable names
==============

We aim to have descriptive names, following PEP8, except for standard quantum numbers,
for which we allow single-letter variable names:

  * ``n``: principal quantum number
  * ``L``: electron orbital angular momentum
  * ``S``: electron spin
  * ``J``: total electronic angular momentum
  * ``I``: nuclear angular momentum
  * ``F``: total (electron + nucleus) angular momentum
  * ``M``: magnetic quantum number. This is a good quantum number at all magnetic
    fields. In the low-field limit ``M = M_F`` while in the high-field limit ``M = M_I + M_J``
  * ``M_J``: electronic magnetic quantum number
  * ``M_I``: nuclear magnetic quantum number

Units
=====

We use SI units throughout this package:

  * Frequencies are in radians per second (``rad/s``)
  * Magnetic fields are in Tesla (``T``)

We define *Rabi frequency*, :math:`\Omega`, such that the pi-time is given by
:math:`t_\pi = \pi / \Omega`.

We define *saturation intensity* so that, for a resonantly-driven cycling
transition, one saturation intensity gives equal stimulated and spontaneous emission rates.

Levels and States
=================

``atomic_physics`` is designed to work at any magnetic field which is small compared with
the spin-orbit coupling. In this case we can use LS coupling and label levels by the
quantum numbers ``n``, ``L``, ``S``, ``J`` and ``I``. These are assumed to be
"good" quantum numbers for all fields supported by the atomic physics package.

``M`` is a good quantum number at all fields however there is generally more than one
state with a given combination of ``(n, L, S, J, I, M)``. 

Since do not make any assumptions about the size of the field compared with the
hyperfine interaction we cannot assume that either (``F``, ``M_F``) or (``M_I``, ``M_J``)
form a basis of good quantum numbers to uniquely identify states by.

Instead, we, uniquely identify states by an index into a list of states ordered by
*decreasing* energy (with index ``0`` being the atom's highest-energy state). This
convention (as opposed to having index ``0`` be the ground state) ensures that the
Pauli operators have their conventional meaning and signs - for example with
:math:`\sigma_+` being the raising operator and transition energies represented by
:math:`H = \frac{1}{2}\omega\sigma_z`.

:class:`atomic_physics.core.Atom` provides a number of helper functions to make working
with state indexes easier. For example, :class:`atomic_physics.core.Atom`
provides helpers to attempt to find the state index corresponding to a given set of
``F``, ``M_F`` (``M_I``, ``M_J``) for cases where the field is low (high) enough for
these to be meaningful. This functionality is implemented by rounding the expectation
values of the relevant operators to the valid value.

The Hamiltonian
---------------

To calculate the energies of each state we diagonalise the Hamiltonian:

.. math::

    H_0 = -\boldsymbol{\mu}\cdot\mathbf{B}
    +A \mathbf{I}\cdot\mathbf{J}
    +B \left(3\left(\mathbf{I}\cdot\mathbf{J}\right)^2 + \frac{3}{2}\mathbf{I}\cdot{\mathbf{J}} - IJ(I+1)(J+1)\right)
    / \left(2IJ(2I - 1)(2J - 1)\right)


where:

* :math:`\boldsymbol{\mu} = - g_J\mu_B\mathbf{J} + g_I\mu_N\mathbf{I}` is the angular
  momentum operator.
* :math:`A` and :math:`B` are the are the hyperfine structure constants for the magnetic
  dipole and electric quadrupole interactions respectively.

High-field basis
----------------

Internally, ``atomic_physics`` often works in the high-field (``M_I``, ``M_J``) basis
where the nuclear and electronic spins are decoupled, which makes calculations easier.
Note that these states are not generally energy eignestates.

We order the states in this basis by increasing ``M_J`` and ``M_I`` *not* by energy ordering.
We order our dimensions as :math:`J \tensor I`. The values of ``M_I`` and ``M_J`` for
each state in this basis are available through :class:`atomic_physics.core.Atom`\'s
`high_field_M_I` and `high_field_M_J` attributes.

Polarizations
=============

We represent polarizations using Jones vectors: 3-element vectors, giving the complex
amplitudes of the relevant field (electric / magnetic, depending on the transition) in
Cartesian coordinates. The field
:math:`\mathbf{A}(t)` is represented by the Jones vector :math:`\mathbf{A}` where
:math:`\mathbf{A}(t) = \Re\left[e^{-iwt} \mathbf{A}\right]`.


Axes
----

The quantisation field is assumed to lie along ``z``. We make no assumptions about the
direction of propagation of the fields.

Jones Matrices
--------------

To help manipulating polarization vectors we provide helper functions, which produce
Jones matrices for a range of polarization transformations. The Jones matrices
are complex-valued 3x3 arrays, which act on the Jones vectors through matrix
multiplication (e.g. using numpy's ``@`` operator). Composite transformations can be
formed by chaining Jones matrices through matrix multiplication (right hand side is
applied first).

The Spherical Basis
-------------------

Internally ``atomic_physics`` often works with polarizations in the spherical basis.
This is a convenient choice because angular momentum operators have simple
representations. Helper functions are provided to convert between the Cartesian and
spherical basis.

The basis vectors are:

.. math::

    \hat{\mathbf{e}}_{-1} &= +\frac{1}{\sqrt{2}}\left(\hat{\mathbf{x}} - i\hat{\mathbf{y}}\right)\\
    \hat{\mathbf{e}}_{+1} &= -\frac{1}{\sqrt{2}}\left(\hat{\mathbf{x}} + i\hat{\mathbf{y}}\right)\\
    \hat{\mathbf{e}_0} &= \hat{\mathbf{z}}

These vectors satisfy orthonormality relations:

.. math::

    \hat{\mathbf{e}}_{\pm 1} \cdot \hat{\mathbf{e}}_{\pm 1} &= 0 \\
    \hat{\mathbf{e}}_{\pm 1} \cdot \hat{\mathbf{e}}_{\mp 1} &= -1 \\
    \hat{\mathbf{e}}_{0} \cdot \hat{\mathbf{e}}_{q} &= \delta_{q, 0}


In the spherical basis, we use the representation:

.. math::

    \mathbf{A} &= \sum_q \left(-1\right)^q A_q \hat{\mathbf{e}}_{-q} \\
    &= \sum_q A_q \hat{\mathbf{e}}_q*\\
    &= -A_{-1} \hat{\mathbf{e}}_{+1} + A_0 \hat{\mathbf{e}}_{0} - A_{+1} \hat{\mathbf{e}}_{-1} 

and represent the vector :math:`\mathbf{A}` by the array
``np.array((A_{-1}, A_0, A_{+1}))``.

The dot product of two vectors in the spherical basis is given by:

.. math::

    \mathbf{A}\cdot\mathbf{B} = \sum_q \left(-1\right)^q A_q B_{-q}

Matrix Elements
===============

We will encounter Hamiltonians of the form:

.. math::

    H = \mathbf{A}(t) \cdot \mathbf{D}

Where :math:`\mathbf{A}(t)` is the (time-dependent) electric or magnetic field vector
and :math:`\mathbf{D}` is some vector-valued operator. For example, the :ref:`mpole`
Hamiltonian is given by :math:`H = - \boldsymbol{\mu} \cdot \mathbf{B}`.

We write the part of the Hamiltonian describing the interaction between two states
:math:`\left|u\right>` and :math:`\left|l\right>`, where :math:`\left|u\right>` is the
state with greater energy (the "upper" state) and :math:`\left|l\right>` is the state
with lower energy, as:

.. math::

    H^{ul} = \left<u|H|l\right> \sigma_+ +
             \left<l|H|u\right> \sigma_- 
             +\frac{1}{2}\left(\left<u|H|u\right> - \left<l|H|l\right>\right) \sigma_z +
             \frac{1}{2}\left(\left<u|H|u\right> + \left<l|H|l\right>\right) \mathbb{1}

where:

* all operators act on the  (u, l) subspace only.
* :math:`\sigma_\pm = \frac{1}{2}\left(\sigma_x \pm i \sigma_{y}\right)`
* for simplicity, we have neglected the additional :math:`\sigma_z` terms arising from
  interactions involving other states.

For now, we will neglect the :math:`\sigma_z` and :math:`\mathbb{1}` terms. These lead to effective
frequency modulation of the drive field and will be treated in the next section.
In this approximation, the Hamiltonian reduces to:

.. math::

    H^{ul} = \left<u|H|l\right> \sigma_+ + \left<l|H|u\right> \sigma_- 

We express the field in terms of its Jones vector, :math:`\mathbf{A}`:

.. math::

    \mathbf{A}(t) &= \Re\left[{\mathbf{A} e^{-i\omega t}}\right]\\
                  &= \frac{1}{2}\left(\mathbf{A} e^{-i \omega t} + \mathbf{A}^* e^{+i \omega t}\right)

Thus:

.. math::

    H^{ul} = \frac{1}{2}\left(
        \left<u|\mathbf{A}\cdot\mathbf{B}|l\right> e^{-i \omega t} +
        \left<u|\mathbf{A}^*\cdot\mathbf{B}|l\right> e^{+i \omega t}
    \right) \sigma_+ +
    \frac{1}{2}\left(
        \left<l|\mathbf{A}\cdot\mathbf{B}|u\right> e^{-i \omega t} +
        \left<l|\mathbf{A}^*\cdot\mathbf{B}|u\right> e^{+i \omega t}
    \right) \sigma_-

Moving into the interaction picture with respect to the atomic Hamiltonian
:math:`H_0 = \frac{1}{2}\omega_0 \sigma_z` this Hamiltonian becomes

.. math::

    H^{ul} &\rightarrow U^\dagger H^{ul} U\\
           &= e^{\frac{1}{2}i\omega_0 t \sigma_z} H^{ul} e^{-\frac{1}{2}i\omega_0 t \sigma_z}

where:

.. math::

    U &:= e^{-i H_0 t}\\
      &= e^{-\frac{1}{2}i\omega_0 t \sigma_z}

from the identity:

.. math::

    e^{ia\left(\hat{\mathbf{n}}\cdot\boldsymbol{\sigma}\right)} = \mathbb{1}\cos{a} + i{\mathbf{n}}\cdot\boldsymbol{\sigma}\sin{a}

it follows that:

.. math::

    U &= \mathbb{1}\cos{\left(-\frac{1}{2}i\omega_0 t\right)}
    + i\sin{\left(-\frac{1}{2}i\omega_0 t\right)}\sigma_z\\
    &=
    \left(\begin{matrix}
    e^{-\frac{1}{2}i\omega_0 t} & 0\\
    0 & e^{+\frac{1}{2}i\omega_0 t}
    \end{matrix}\right)
    \\

Multiplying through, we find that:

.. math::

    U^\dagger \sigma_\pm U = e^{\pm i\omega_0 t} \sigma_\pm

Substituting into our Hamiltonian gives:

.. math::
    H^{ul} &= \frac{1}{2} \left(
        \left<u|\mathbf{A}\cdot\mathbf{B}|l\right> e^{-i \omega t} +
        \left<u|\mathbf{A}^*\cdot\mathbf{B}|l\right> e^{+i \omega t}
    \right) e^{i \omega_0 t}\sigma_+\\
    &
    + \frac{1}{2}\left(
        \left<l|\mathbf{A}\cdot\mathbf{B}|u\right> e^{-i \omega t} +
        \left<l|\mathbf{A}^*\cdot\mathbf{B}|u\right> e^{+i \omega t}
    \right) e^{-i \omega_0 t}\sigma_-
    \\
    &= \frac{1}{2}\left<u|\mathbf{A}\cdot\mathbf{B}|l\right> e^{i (\omega_0 - \omega) t} \sigma_+ \\
    & + \frac{1}{2}\left<u|\mathbf{A}^*\cdot\mathbf{B}|l\right> e^{i (\omega_0 + \omega) t} \sigma_+ \\
    & + \frac{1}{2} \left<l|\mathbf{A}\cdot\mathbf{B}|u\right> e^{-i (\omega + \omega_0) t} \sigma_- \\
    & + \frac{1}{2}\left<l|\mathbf{A}^*\cdot\mathbf{B}|u\right> e^{-i (\omega_0 - \omega) t} \sigma_- \\
    &= \frac{1}{2}\left<u|\mathbf{A}\cdot\mathbf{B}|l\right> e^{-i \delta t} \sigma_+ \\
    & + \frac{1}{2}\left<u|\mathbf{A}^*\cdot\mathbf{B}|l\right> e^{i(2\omega_0 + \delta) \delta t} \sigma_+ \\
    & + \frac{1}{2} \left<l|\mathbf{A}\cdot\mathbf{B}|u\right> e^{-i(2\omega_0 + delta) t} \sigma_- \\
    & + \frac{1}{2}\left<l|\mathbf{A}^*\cdot\mathbf{B}|u\right> e^{+i \delta t} \sigma_- \\

where we have defined the detuning :math:`\omega = \omega_0 + \delta`.

Making a rotating wave approximation, dropping the counter-rotating terms, results
in the standard Rabi flopping Hamiltonian:

.. math::

    H^{ul} &=
        \frac{1}{2}\left<u|\mathbf{A}\cdot\mathbf{B}|l\right> e^{-i \delta t} \sigma_+ +
        & + \frac{1}{2}\left<l|\mathbf{A}^*\cdot\mathbf{B}|u\right> e^{+i \delta t} \sigma_-\\
    &= \frac{1}{2}\Omega e^{-i \delta t} \sigma_+ + \mathrm{hc}

where ":math:`\mathrm{hc}`" denotes the hermitian conjugate and the Rabi frequency is
given by:

.. math::

    \Omega = \left<u|\mathbf{A}\cdot\mathbf{B}|l\right>

.. _fm_mod:

Frequency Modulation by the Drive Field
---------------------------------------

We now come back to the :math:`\sigma_z` terms we neglected in the previous section (the
:math:`\mathbb{1}` terms turn into effective :math:`\sigma_z` terms between different
paris of states) we have:

.. math::

    H^z &= \frac{1}{2}\left(\left<u|H|u\right> - \left<l|H|l\right>\right) \sigma_z \\

This Hamiltonian is unchanged by moving into the interaction picture with respect to
:math:`{H_0}`. Expanding the form of the Hamiltonian we have

.. math::

    H^z &= \frac{1}{2}\left(\left<u|H|u\right> - \left<l|H|l\right>\right) \sigma_z\\
        &= \frac{1}{4}\left(\mathbf{A} e^{-i \omega t} + \mathbf{A}^*e^{+i \omega t}\right)
        \cdot\left(\left<u|\mathrm{B}|u\right> - \left<l|\mathrm{B}|l\right> \right) \sigma_z\\

When we move the remainder of :math:`H^{ul}` into the interaction picture with respect to
this Hamiltonian we end up with time dependencies like
:math:`e^{i\left(\delta + \alpha\cos\omega\right)t}`, which are equivalent to frequency modulation
of our RF drive at the RF drive frequency.

*We will generally ignores this effect, assuming that the modulation depth
is sufficiently small for these terms to be negligible*. However, if the modulation depth
becomes non-negligible these terms will affect the dynamics and must be factored in.

.. _mpole:

Magnetic Dipole Interaction
===========================

The magnetic dipole Hamiltonian is:

.. math::

    H = - \boldsymbol{\mu} \cdot \mathbf{B}

We wish to find the Rabi frequency:

.. math::

    \Omega &= -\left<u|\boldsymbol{\mu}\cdot\mathbf{B}|l\right> \\
    &= \sum_q \left(-1\right)^{q+1} B_{-q} \left<u|\mu_q|l\right>

The angular momentum operator is given by:

.. math::

    \boldsymbol{\mu} = \mu_N g_I \mathbf{I} - \mu_B g_J \mathbf{J}

We will work in the high-field (:math:`\left|I, J, M_I, M_J\right>`) basis where the
nuclear and electron angular momenta are decoupled. This allows us to consider the two
angular momenta separately.

To evaluate the matrix element, we need to know the elements of the angular momentum
operator in the spherical basis. These are related to the "ladder operators", 
:math:`J_\pm`, by :math:`J_{\pm 1} = \mp \frac{1}{\sqrt{2}}J_\pm` and similarly for :math:`I`.

We thus have:

.. math::

    J_{\pm 1} \left|J, M_J\right> &= \mp \hbar \frac{1}{\sqrt{2}} \sqrt{(J \mp M_J ) (J \pm M_J + 1)}\left|J, M_J\pm1\right>\\
    J_0 \left|J, M_J\right> &= \hbar M_J\left|J, M_J\right>

It follows that:

.. math::

    \left<M_J=n | \mu_q | M_J = m\right> \propto \delta\left(n, m + q\right)

so:

.. math::

    \Omega &= \sum_q (-1)^{q+1} B_{-q} \left<u|\mu_q|l\right> \delta\left(M_u, M_l + q\right)\\
           &= R_{ul} B_{-q}

where: :math:`R_{ul} := (-1)^{q+1}\left<u|\mu_q|l\right>` and :math:`q = M_u - M_l`.
We will refer to :math:`R_{ul}` as the "magnetic dipole matrix element".

Selection Rules
---------------

From the above, it follows that:

    * The field :math:`\mathbf{B} = -B_{-1} \hat{\mathbf{e}}_{+1}` drives only :math:`\sigma_+` transitions, for which :math:`M_u - M_l = +1`.
    * The field :math:`\mathbf{B} = -B_{+1} \hat{\mathbf{e}}_{-1}` drives only :math:`\sigma_-` transitions, for which :math:`M_u - M_l = -1`.
    * The field :math:`\mathbf{B} = B_{0} \hat{\mathbf{e}}_{0}` drives only :math:`\pi` transitions, for which :math:`M_u = M_l`.

.. _rates:

Rate Equations
==============

Rate equations describe the evolution of state populations resulting from the interaction
between an atom and a set of laser beams, neglecting the impact of coherent interactions
between different transitions.

We describe the atom's state at time :math:`t` by the population vector
:math:`\mathbf{v}(t)`, which gives the probabilities of the atom being in each state
at time :math:`t` (as usual, :math:`\mathbf{v}(t)_{-1}` is the ground-state probability
and :math:`\mathbf{v}(t)_0` is the probability for the highest-energy state).

The *transitions matrix*, :math:`T`, describes the evolution of the state population
vector over time:

.. math::
    
    \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = T \mathbf{v}(t)

Note that :math:`T_{i, j}\mathbf{v}_j` gives the rate of population transfer from state
:math:`j` to state :math:`i`.

Assuming the laser properties (detuning, intensity, polarization) do not change with
time, this differential equation can be solved to get:

.. math::

   \mathbf{v}(t)  = e^{T t} \mathbf{v}(t=0)

The transition matrix is formed from two components: the *stimulated emissions* matrix,
which describes the interaction between the atom and the laser fields; and,
the *spontaneous emissions* matrix, which describes the atom's behaviour in the absence
of any applied lasers.

Note that, since :math:`T` is a matrix, it should be exponentiated using ``numpy``'s
``expm`` function.

Steady State
------------

For cases where all states which are reachable by the atom can decay to the ground state
(i.e. there are no "dark states" which the atom can get stuck in),
the steady-state solution (:math:`t \rightarrow \infty`) is given by the solution to
the equation:

.. math::

    \frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} &= 0\\
    T  \mathbf{v}\left(t\rightarrow\infty\right) &= 0

subject to the constraint that :math:`\sum_i \mathbf{v}\left(t\rightarrow\infty\right)_i = 1`
(i.e. we don't want the trivial solution where there is no population in any state!).

We impose this constraint by converting the above to the equation:

.. math::

    T' \mathbf{v}\left(t\rightarrow\infty\right) = \mathbf{a}

where:

.. math::

    T'_{i, j} &= \left\{ \begin{matrix}
        T_{ij} & i \neq 0 \\
        1 & i = 0
    \end{matrix}\right. \\

    \mathbf{a}_i  &= \left\{ \begin{matrix}
        0 & i \neq 0 \\
        1 & i = 0
    \end{matrix}\right. \\

NB no information is lost by removing the first row of :math:`T` because it is
rank-deficient, with only :math:`N - 1` linearly independent rows for an atom with
:math:`N` states (the transition rate out of any state must be equal to the
sum of the rates of transitions from that state into all other states).