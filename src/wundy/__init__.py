"""
WUNDY: 1D Finite Element Solver (Linear + Nonlinear)

This package provides:

- YAML-based input
- Preprocessing of nodes, elements, BCs, loads, materials
- Element-level routines (stiffness, internal forces, loads)
- Nonlinear Newton-Raphson solver
- High-level `solve()` function for end users

Typical usage:

    import wundy
    result = wundy.solve("my_case.yaml")
"""

from .ui import load, preprocess, solve
from .elements import (
    t1d1_element_stiffness,
    t1d1_element_internal_force,
    t1d1_element_uniform_load,
)
from .materials import (
    linear_elastic_stress,
    linear_elastic_tangent,
)
from .solver import newton_solve_1d

__all__ = [
    # High-level API
    "solve",
    "load",
    "preprocess",

    # Solver
    "newton_solve_1d",

    # Elements
    "t1d1_element_stiffness",
    "t1d1_element_internal_force",
    "t1d1_element_uniform_load",

    # Materials
    "linear_elastic_stress",
    "linear_elastic_tangent",
]
