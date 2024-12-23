.. _testing:

Testing methodology
===================

``ionics-fits`` is heavily tested using both unit tests and fuzzing.

Unit Tests
~~~~~~~~~~

* run using ``poe test``
* to run a subset of tests use the ``-k`` flag e.g. ``poe test -k "rabi"`` to only run
  tests with the word "rabi" in their name. For more information about configuring
  pytest see the `documentation <https://docs.pytest.org/en/7.1.x/>`_
* all tests must pass before a PR can be merged into master as well as static analysis
  based on `pytype`, run using ``poe types`` (Linux only)
* PRs to add new models will only be merged once they have full test coverage
* unit tests aim to provide good coverage over the space of "reasonable datasets". There
  will always be corner-cases where the fits fail and that's fine; the aim here is to
  cover the main cases users will hit in the wild
* when a user hits a case in the wild where the fit fails unexpectedly (i.e. we think
  the fit code should have handled it), a `regression test` based on the failing
  dataset should be added
* unit tests should be deterministic. Synthetic datasets should be included in the test
  rather than randomly generated at run time. Tip: while writing a test it's fine to let
  the test code generate datasets for you. Once you're happy, run the test in verbose
  mode and copy the dataset from the log output

Fuzzing
~~~~~~~~~~

* fuzzing is non-deterministic (random parameter values, randomly generated datasets)
  exploration of the parameter space.
* used when developing / debugging fits, but not automatically run by CI
* run with ``poe fuzz`` (see ``--help`` for details)
* fit failures during fuzzing are not automatically considered a bug; unlike unit tests,
  fuzzing explores the "unreasonable" part of parameter space as well. Indeed, a large
  part of the point of fuzzing is to help the developer understand what should be
  considered "reasonable" (this information should end up in the documentation for the
  fuzzed model).
* fuzzing is considered a tool to help developers finding corner-cases. We don't aim
  for the same level of code coverage with fuzzing that we do with the unit tests which
  *should*, for example, cover ever code path in every parameter estimator
