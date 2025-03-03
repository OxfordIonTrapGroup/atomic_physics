import sys
from juliacall import Main as jl


def fmt():
    jl.seval("using JuliaFormatter")
    jl.seval('format(".")')


def fmt_test():
    jl.seval("using JuliaFormatter")
    ret = jl.seval('format(".")')

    if not ret:
        sys.exit(1)
