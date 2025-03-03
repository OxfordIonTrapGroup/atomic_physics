.. _changes:

Change Log
==========

v2.1
~~~~

* ``Atom.intensity_to_power`` was renamed to ``Atom.intensity_for_power`` for consistency with other methods.
* ``Atom.get_rabi_rf`` now takes a tuple of state indices because it does not actually care which is the lower and which is the upper.
* new ``Atom.get_amplitude_for_rabi_rf`` method

v2.0.3
~~~~~~

Refactor of how we define ions and atoms. We now define a subclass of ``AtomFactory`` for each atom / ion definition. Pre-defined levels are now attributes of these classes. This change is partly motivated by making it easier to write species-agnostic code (makes it easier to write things like ``factory.S12`` rather than having to access module-level attributes as well as the factory object).

v2.0
~~~~

This version gives the ``atomic_physics`` API a much needed tidy up.

A major design goal for the tidy up was to make things more obvious for the user. This
comes at the expense of some function names now being quite a bit more verbose, but
that feels like a worthwhile price to pay for clarity!

Misc:

* We now have a documentation build. Closes `#32 <https://github.com/OxfordIonTrapGroup/atomic_physics/issues/32>`_
* States are now ordered in *decreasing* energy order, not increasing! This change allows
  the Pauli matrices to take on their customary meanings and signs, for example with
  :math:`\sigma_+` being the raising operator and energies being represented by
  :math:`+\frac{1}{2}\omega\sigma_z`.
* Formatting and linting moved from flake8 and black to ruff
* CI now checks type annotations using pytype
* Fix assorted type annotation and docstring bugs
* Significantly expanded test coverage
* Significantly expanded documentation
* Added helper functions to convert between different polarization representations
* Added a helper function to calculate the Rayleigh range of a beam
* Added a new ``polarizations`` module for representing and manipulating polarizations.
* Added a new ``RFDrive`` class. This is a bit heavyweight for just calculating
  AC Zeeman shifts (which is all we use it for at present) but it is mainly intended
  for a future optical bloch equations solver.
* Added a simple ``TwoStateAtom`` class to help making simple tests and simulations.

Bug fixes:

* Calculate derivatives properly in transition sensitivity calculations. Closes
  `#24 <https://github.com/OxfordIonTrapGroup/atomic_physics/issues/24>`_
* Fix indexing in AC Zeeman shift calculation. Closes
  `#78 <https://github.com/OxfordIonTrapGroup/atomic_physics/issues/78>`_
* Fix incorrect transition frequencies for calcium

API refactor:

* Named tuples have been replaced with data classes
* We no longer export classes at the module level. Replace `import atomic_physics as ap`
  with `from atomic_physics.core import Atom`
* General push to avoid "partially constructed objects" - i.e. objects where we
  can't set all the fields at construction time so rely on mutating them in
  non-obvious ways over the object's lifetime. This makes the code easier to follow
  and removes the need for a bunch of checks to see if fields have been initialised.
* General push to make variable and function names more explicit, even at the cost
  of increased verbosity (optimizing for least surprise not fewest keystrokes!).
  Closes `#30 <https://github.com/OxfordIonTrapGroup/atomic_physics/issues/30>`_
* ``LevelData`` now only contains the atomic structure data; information about the
  energy-ordering of states is now in a separate ``LevelStates`` object.
* ``Atom.slice`` has been renamed to ``Atom.get_slice_for_level``. This avoids shadowing the name of a built-in python type.
* ``Atom.detuning`` has been renamed to ``Atom.get_transition_frequency_for_states``. This method
  supports an additional ``relative`` keyword to calculate absolute transition
  frequencies. Closes
  `#29 <https://github.com/OxfordIonTrapGroup/atomic_physics/issues/29>`_
* ``Atom.index`` has been split into ``get_states_for_M``, ``get_state_for_F``,
  ``get_state_for_MI_MJ``. This avoids having one function which does lots of
  different jobs and has a return type which depends in non-obvious ways on the
  input parameters.
* ``Atom.level`` has been renamed ``get_level_for_state``
* added a new ``Atom.get_transition_for_levels`` helper function
* ``Atom.population`` has been removed as it wasn't particularly useful
* ``Atom.I0`` has been renamed ``Atom.get_saturation_intensity``
* ``Atom.P0`` has been renamed ``intensity_for_power``
* ``Laser.q`` has been renamed to ``Laser.polarization``
* ``Laser.I`` has been renamed to ``Laser.intensity``
* ``Atom.B`` has been renamed to ``Atom.magnetic_field``
* ``Atom.I`` has been renamed to ``Atom.nuclear_spin``
* ``Laser.delta`` has been renamed to ``Laser.detuning``
* ``RateEquations.get_spont`` has been renamed to ``RateEquations.get_spont_matrix``.
* ``RateEquations.get_stim`` has been renamed to ``RateEquations.get_stim_matrix``.
* ``RateEquations.get_transitions`` has been renamed to ``RateEquations.get_transitions_matrix``.
* ``RateEquations.steady_state`` has been renamed to ``RateEquations.get_steady_state_populations``.
* ``RateEquations.get_steady_state_populations`` now only takes a transitions matrix
  as an input, not a transitions matrix or a set of lasers (supporting multiple input
  types saved a little boiler plate at the expense of making things complex/confusing).
* angular frequencies are denoted ``w`` not ``f``
* Methods where we don't care about the orderings of states / levels now take a
  tuple of states / levels rather than asking for an "upper" and "lower" one
* The AC Zeeman shift code now takes a ``RFDrive`` object
* Polarizations are new represented by Jones vectors rather than +-1 or 0. The old
  system was relatively easy once one understood it, but only worked for the simple
  case of rate equations. Anticipating doing more complex things like optical bloch
  equations, I've started moving us over to Jones vectors.
* Add ``operators.expectation_value`` helper method.
* Add ``Atom.levels`` field.
* Add new ``Atom.get_states_for_level`` method.
* ``utils.field_insensitive_point`` now works with values of ``F`` and ``M_F`` instead of state indices.
