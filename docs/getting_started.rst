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

As an example, we use the rate equations interface to simulate optical pumping from the
ground-level ``F=4, M=+4`` "stretched state" to the 3D5/2 level using a 393nm laser.

.. code-block:: python
   :linenos:

   import numpy as np
   from scipy.linalg import expm
   import matplotlib.pyplot as plt

   import atomic_physics as ap
   from atomic_physics.ions import ca43


   t_ax = np.linspace(0, 100e-6, 100)

   ion = ca43.Ca43(B=146e-4)
   stretch = ion.get_index(ca43.ground_level, 4)

   rates = ap.rates.Rates(ion)
   delta = ion.get_transition_frequency(stretch, ion.get_index(ca43.P32, +5))
   lasers = [ap.Laser("393", q=+1, I=0.02, delta=delta)]  # resonant 393 sigma+
   trans = rates.get_transitions(lasers)

   Vi = np.zeros((ion.num_states, 1))  # initial state
   Vi[stretch] = 1  # start in F=4, M=+4
   shelved = np.zeros(len(t_ax))
   for idx, t in np.ndenumerate(t_ax):
      Vf = expm(trans * t) @ Vi
      shelved[idx] = sum(Vf[ion.get_slice(ca43.shelf)])

   plt.plot(t_ax * 1e6, shelved)
   plt.ylabel("Shelved Population")
   plt.xlabel("Shelving time (us)")
   plt.grid()
   plt.show()

