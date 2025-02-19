.. _getting_started:

Getting Started
===============

Installation
~~~~~~~~~~~~

Install from `pypi` with

.. code-block:: bash

   pip install atomic_physics

or add to your poetry project with

.. code-block:: bash

   poetry add atomic_physics

Basic usage
~~~~~~~~~~~

The heart of the ``atomic_physics`` package is the :class:`~atomic_physics.core.Atom`,
which represents a particular atomic species at a given magnetic field.

:class:`~atomic_physics.core.AtomFactory` provides a flexible means of constructing
:class:`~atomic_physics.core.Atom`\s. A number of 
:class:`~atomic_physics.core.AtomFactory`\s are available in the :ref:`atoms` and
:ref:`ions` modules, providing pre-configured atom definitions based on accurate atomic
physics data.

The :class:`~atomic_physics.rate_equations.Rates` class provides a simple interface
for performing rate equations simulations.

Example Usage
~~~~~~~~~~~~~

As an example, we use the rate equations interface to simulate electron shelving in
43Ca+ - optical pumping from the ground-level ``F=4, M=+4`` "stretched state" to
the 3D5/2 level using a 393nm laser.

.. testcode::

   """Simple rate equations example of 393 shelving in 43Ca+."""

   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.linalg import expm

   from atomic_physics.core import Laser
   from atomic_physics.ions.ca43 import Ca43
   from atomic_physics.rate_equations import Rates

   t_ax = np.linspace(0, 100e-6, 100)
   intensity = 0.02  # 393 intensity

   ion = Ca43(magnetic_field=146e-4)
   stretch = ion.get_state_for_F(Ca43.ground_level, F=4, M_F=+4)

   rates = Rates(ion)
   delta = ion.get_transition_frequency_for_states(
       (stretch, ion.get_state_for_F(Ca43.P32, F=5, M_F=+5))
   )
   lasers = (
       Laser("393", polarization=+1, intensity=intensity, detuning=delta),
   )  # resonant 393 sigma+
   trans = rates.get_transitions_matrix(lasers)

   Vi = np.zeros((ion.num_states, 1))  # initial state
   Vi[stretch] = 1  # start in F=4, M=+4
   shelved = np.zeros(len(t_ax))
   for idx, t in np.ndenumerate(t_ax):
       Vf = expm(trans * t) @ Vi
       shelved[idx] = np.sum(Vf[ion.get_slice_for_level(Ca43.shelf)])

   plt.plot(t_ax * 1e6, shelved)
   plt.ylabel("Shelved Population")
   plt.xlabel("Shelving time (us)")
   plt.grid()
   plt.show()
