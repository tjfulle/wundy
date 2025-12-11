import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import wundy
from wundy import first


def polynomial_load_fn(load_expr: str):
    """Create a vectorized callable ``q(x)`` from a polynomial expression string."""

    def _load(x: np.ndarray | float) -> np.ndarray:
        return eval(load_expr, {"x": x, "np": np})

    return _load


def equivalent_beam_load(load_expr: str, x1: float, x2: float, n_gauss: int = 3) -> np.ndarray:
    """Compute consistent nodal forces for a polynomial transverse load on a beam."""

    q = polynomial_load_fn(load_expr)
    length = x2 - x1
    pts, wts = first.gauss_points_weights(n_gauss)

    f_e = np.zeros(4)
    for r, wt in zip(pts, wts):
        # Map parent coordinate to physical space.
        x = 0.5 * ((1.0 - r) * x1 + (1.0 + r) * x2)
        n = first.beam_shape_functions(np.array([r]), length).flatten()
        f_e += n * q(x) * (length / 2.0) * wt

    return f_e


def test_elem_ext_force_matches_dense_integration():
    """Element load vector for an equation-based load matches dense integration."""

    load_expr = "2.0 + 5.0 * x + 3.0 * x**2"
    x1, x2 = 0.0, 0.8
    xe = np.array([[x1], [x2]])
    props = {"type": "EULER", "properties": {"inertia": 1.0, "area": 1.0}}
    material = {"type": "ELASTIC", "parameters": {"E": 1.0}}
    dload = {"type": "BX", "value": load_expr, "direction": [1.0], "elements": [0], "n_gauss": 3}

    gauss_load = first.elem_ext_force(props, material, xe, dload)

    q = polynomial_load_fn(load_expr)
    length = x2 - x1
    xs = np.linspace(x1, x2, 2001)
    rs = 2.0 * (xs - x1) / length - 1.0
    shapes = first.beam_shape_functions(rs, length)
    q_vals = q(xs)
    dense_load = np.array([np.trapz(shapes[i] * q_vals, xs) for i in range(4)])

    assert np.allclose(gauss_load, dense_load, rtol=1e-6, atol=1e-9)


def test_manufactured_solution_polynomial_load():
    """The FE solution should match the manufactured polynomial beam solution."""

    # Manufactured displacement and slope fields (zero at the boundaries).
    def w_exact(x: np.ndarray) -> np.ndarray:
        return x**3 - 3.0 * x**4 + 3.0 * x**5 - x**6

    def theta_exact(x: np.ndarray) -> np.ndarray:
        return 3.0 * x**2 - 12.0 * x**3 + 15.0 * x**4 - 6.0 * x**5

    q_expr = "-72.0 + 360.0 * x - 360.0 * x**2"

    e_modulus = 1.0
    inertia = 1.0
    element_props = {"properties": {"inertia": inertia, "area": 1.0}}
    material = {"type": "ELASTIC", "parameters": {"E": e_modulus}}

    length = 1.0
    n_elem = 8
    result = first.manufactured_solution_beam(
        length=length,
        n_elem=n_elem,
        q_expr=q_expr,
        w_exact=w_exact,
        theta_exact=theta_exact,
        material=material,
        element_props=element_props,
        n_gauss=3,
    )

    assert result["solution"]["converged"]
    assert np.allclose(result["displacements"], result["exact_displacements"], atol=1e-6)
    assert np.allclose(result["rotations"], result["exact_rotations"], atol=1e-5)
