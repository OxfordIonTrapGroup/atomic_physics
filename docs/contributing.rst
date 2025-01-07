Contributing
============

Pull requests are always welcomed. To make reviews fast and easy, before opening a PR,
please:

* Please make sure that all new code is thoroughly tested (see existing tests for inspiration)!
* Please make sure all new code is well documented, including any useful additions to :ref:`definitions`.
* Update the package version number in ``pyproject.toml``
* Update the documentation and :ref:`changes` as required
* Check formatting: ``poe fmt``
* Run lints: ``poe lint --fix``
* Run test suite: ``poe test``
* Check type annotations (Linux only): ``poe types``
* Check the documentation builds correctly (Linux only): ``poe docs``

