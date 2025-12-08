"""
elements.py
-----------
1D two-node bar / truss element (T1D1) routines.

This module provides element-level operations required by the tests:

    t1d1_element_stiffness(x_e, A, material, ngauss=2)
    t1d1_element_internal_force(x_e, u_e, A, material, ngauss=2)
    t1d1_element_uniform_load(x_e, A, q, direction, ngauss=2)

where:
    x_e       : array-like of shape (2,), nodal coordinates [x1, x2]
    u_e       : array-like of shape (2,), nodal displacements [u1, u2]
    A         : cross-sectional area (float)
    material  : material dictionary, e.g.
                {
                    "type": "ELASTIC",
                    "name": "STEEL",
                    "parameters": {"E": 100.0, "nu": 0.3},
                    "density": 1.0,
                }
    q         : uniform distributed load (per unit length)
    direction : scalar factor (+1 or -1 for axial direction)
    ngauss    : number of Gauss points (1 or 2)

The tests assume linear elasticity, but the implementation is compatible
with nonlinear materials via the dispatchers in materials.py.
"""

from __future__ import annotations

import numpy as np

from .materials import get_material_stress, get_material_tangent


# ----------------------------------------------------------------------
# Gauss–Legendre quadrature helper
# ----------------------------------------------------------------------


def _gauss_legendre_1d(ngauss: int):
    """
    Return Gauss–Legendre points and weights on [-1, 1].

    Supports ngauss = 1 or 2 (sufficient for our linear 1D bar).
    """
    if ngauss == 1:
        pts = np.array([0.0])
        wts = np.array([2.0])
    elif ngauss == 2:
        a = 1.0 / np.sqrt(3.0)
        pts = np.array([-a, a])
        wts = np.array([1.0, 1.0])
    else:
        raise ValueError(f"Unsupported number of Gauss points: {ngauss}")
    return pts, wts


# ----------------------------------------------------------------------
# Shape functions and kinematics
# ----------------------------------------------------------------------


def _element_length(x_e: np.ndarray) -> float:
    x_e = np.asarray(x_e, dtype=float).ravel()
    if x_e.size != 2:
        raise ValueError(f"T1D1 expects 2 nodal coordinates, got {x_e}")
    L = x_e[1] - x_e[0]
    if L <= 0.0:
        raise ValueError(f"Element length must be positive, got L={L}")
    return L


def _shape_functions(ksi: float):
    """
    Linear 2-node shape functions N1, N2 at natural coordinate ksi ∈ [-1, 1].
    """
    N1 = 0.5 * (1.0 - ksi)
    N2 = 0.5 * (1.0 + ksi)
    return np.array([N1, N2])


def _B_matrix(L: float):
    """
    Strain–displacement "matrix" B for 1D bar:

        ε = B * u_e

    For a two-node element:

        B = [-1/L, 1/L]
    """
    return np.array([-1.0 / L, 1.0 / L])


# ----------------------------------------------------------------------
# Public element routines
# ----------------------------------------------------------------------


def t1d1_element_stiffness(
    x_e,
    A: float,
    material: dict,
    ngauss: int = 2,
) -> np.ndarray:
    """
    Element stiffness matrix for a 2-node 1D bar element.

    For linear elasticity, the closed form is:

        k_e = (E * A / L) * [[ 1, -1],
                             [-1,  1]]

    This function is implemented via Gauss integration so that it can
    also be used with nonlinear tangent moduli if needed.

    Parameters
    ----------
    x_e : array-like of length 2
        Nodal coordinates [x1, x2].
    A : float
        Cross-sectional area.
    material : dict
        Material dictionary with at least parameters['E'].
    ngauss : int
        Number of Gauss points (1 or 2).

    Returns
    -------
    ke : (2, 2) ndarray
        Element stiffness matrix.
    """
    x_e = np.asarray(x_e, dtype=float).ravel()
    L = _element_length(x_e)
    A = float(A)

    # For linear elasticity, tangent is constant; we evaluate at strain = 0.
    # For nonlinear materials, a more advanced routine could pass in strain.
    Et = get_material_tangent(material, strain=0.0)

    ke = np.zeros((2, 2), dtype=float)
    B = _B_matrix(L)

    ksi_pts, wts = _gauss_legendre_1d(ngauss)
    J = L / 2.0

    for w in wts:
        # contribution: B^T * Et * A * B * J * w
        ke += np.outer(B, B) * (Et * A * J * w)

    return ke


def t1d1_element_internal_force(
    x_e,
    u_e,
    A: float,
    material: dict,
    ngauss: int = 2,
) -> np.ndarray:
    """
    Internal force vector for a 2-node 1D bar element.

    Definition:

        f_int = ∫ B^T * σ(ε) * A dx

    For linear elasticity with constant strain, this reduces to:

        f_int = [ -σ A,
                   σ A ]

    where σ = E * (u2 - u1) / L.

    Parameters
    ----------
    x_e : array-like of length 2
        Nodal coordinates [x1, x2].
    u_e : array-like of length 2
        Nodal displacements [u1, u2].
    A : float
        Cross-sectional area.
    material : dict
        Material dictionary.
    ngauss : int
        Number of Gauss points.

    Returns
    -------
    f_int : (2,) ndarray
        Element internal force vector.
    """
    x_e = np.asarray(x_e, dtype=float).ravel()
    u_e = np.asarray(u_e, dtype=float).ravel()
    if u_e.size != 2:
        raise ValueError(f"T1D1 expects 2 nodal displacements, got {u_e}")

    L = _element_length(x_e)
    A = float(A)

    B = _B_matrix(L)
    # Constant strain in a linear 2-node bar:
    strain = float(B @ u_e)

    sigma = get_material_stress(material, strain)

    f_int = np.zeros(2, dtype=float)
    ksi_pts, wts = _gauss_legendre_1d(ngauss)
    J = L / 2.0

    for w in wts:
        # contribution: B^T * σ * A * J * w
        f_int += B * (sigma * A * J * w)

    return f_int


def t1d1_element_uniform_load(
    x_e,
    A: float,
    q: float,
    direction,
    ngauss: int = 2,
) -> np.ndarray:
    """
    Consistent nodal load vector for a uniform distributed load.

    Definition:

        f_ext = ∫ N^T * q * A * dir dx

    For a constant q and unit area, the classic result is:

        f_ext = [ q L / 2,
                  q L / 2 ]

    (as checked in test_t1d1_element_uniform_load_matches_classic_result).

    Parameters
    ----------
    x_e : array-like of length 2
        Nodal coordinates [x1, x2].
    A : float
        Cross-sectional area (kept for generality).
    q : float
        Uniform load intensity (per unit length).
    direction : float or array-like
        Direction factor; scalar in axial problems.
    ngauss : int
        Number of Gauss points.

    Returns
    -------
    f_ext : (2,) ndarray
        Element equivalent nodal force vector.
    """
    x_e = np.asarray(x_e, dtype=float).ravel()
    L = _element_length(x_e)
    A = float(A)
    q = float(q)

    # direction may be passed as scalar or list; reduce to scalar
    if isinstance(direction, (list, tuple, np.ndarray)):
        if len(direction) == 0:
            raise ValueError("direction must not be empty")
        direction_scalar = float(direction[0])
    else:
        direction_scalar = float(direction)

    f_ext = np.zeros(2, dtype=float)

    ksi_pts, wts = _gauss_legendre_1d(ngauss)
    J = L / 2.0

    for ksi, w in zip(ksi_pts, wts):
        N = _shape_functions(ksi)  # [N1, N2]
        # contribution: N^T * q * A * direction * J * w
        f_ext += N * (q * A * direction_scalar * J * w)

    return f_ext
