# import unittest

# import numpy as np


# import line_profiler


# @line_profiler.profile
# def main():
#     import juliacall
#     from juliacall import Main as jl, convert as jlconvert

#     # jl.seval("using QuantumOptics")
#     jl.seval("using bloch")

#     t_in = np.linspace(0, 1, 100)
#     t_out, psit = jl.bloch.bloch.solve_obe(t_in)
#     t_out, psit = jl.bloch.solve_obe(t_in)

#     t_jl = jlconvert(jl.AbstractVector[jl.Float64], t_in)
#     t_jl = jlconvert(jl.AbstractVector[jl.Float64], t_in)
#     t_out, psit = jl.bloch.solve_obe(t_jl)
#     t_out, psit = jl.bloch.solve_obe(t_jl)

#     t_out_py = np.array(t_out)
#     t_out_py = np.array(t_out)
#     Ps = np.array([np.abs(psi[0]) ** 2 for psi in psit])

#     # class Bloch(unittest.TestCase):
#     #     def test_rf_drive(self):
#     #         """
#     #         Test the optical Bloch equations for a 2-state system driven by an RF field.
#     #         """
#     #         ion = ca40.Ca40.filter_levels(level_filter=[ca40.S12])(magnetic_field=146.0942e-4)

#     #         t = jlconvert(jl.AbstractVector[jl.Float64], np.linspace(0, 1, 100))
#     #         t, psit = jl.solve_obe(t)

#     #         t = np.array(t)
#     #         Ps = np.array([np.abs(psi[0])**2 for psi in psit])


# main()
