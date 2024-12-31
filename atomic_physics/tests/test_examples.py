import unittest

from matplotlib import pyplot as plt


def dummy_show():
    pass


plt.show = dummy_show


class TestExamples(unittest.TestCase):
    def test_examples(self):
        """Check that the example code runs without error."""
        import atomic_physics.examples.breit_rabi  # noqa: F401
        import atomic_physics.examples.ca_shelving  # noqa: F401
        import atomic_physics.examples.clock_qubits  # noqa: F401
        import atomic_physics.examples.M1_transitions  # noqa: F401
