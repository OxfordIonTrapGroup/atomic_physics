from juliacall import Main as jl


def fmt():
    jl.seval("using JuliaFormatter")
    jl.seval('format(".")')
