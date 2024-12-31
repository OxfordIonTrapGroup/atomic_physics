# """Test Spin 1/2 nuclei"""
# import unittest
# from atomic_physics.ions import ca40


# class TesAtomFactory(unittest.TestCase):
#     def test_sorting(self):
#         ion = ca40.Ca40.filter_levels(level_filter=(ca40.ground_level, ca40.shelf))(magnetic_field=100e-4)

#         # check that states with indices 0 and 1 belong to S1/2
#         for i in range(2):
#             self.assertEqual(ion.get_level_for_state(i).L, 0)
#             self.assertEqual(ion.get_level_for_state(i).J, 0.5)

#         # check that states with indices 2-7 belong to D5/2
#         for i in range(6):
#             self.assertEqual(ion.get_level_for_state(i + 2).L, 2)
#             self.assertEqual(ion.get_level_for_state(i + 2).J, 2.5)
