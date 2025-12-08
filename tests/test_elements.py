import numpy as np
import pytest

from wundy.elements import (
    t1d1_element_stiffness,
    t1d1_element_internal_force,
    t1d1_element_uniform_load,
)


def _test_material(E: float = 100.0):
    return {
        "type": "ELASTIC",
        "name": "STEEL",
        "parameters": {"E": E, "nu": 0.3},
        "density": 1.0,
    }


def test_t1d1_element_stiffness_matches_closed_form():
    L = 2.0
    A = 1.0
    E = 100.0
    x_e = np.array([0.0, L])

    ke = t1d1_element_stiffness(x_e, A, _test_material(E), ngauss=2)

    ke_exact = (E * A / L) * np.array([[1.0, -1.0],
                                       [-1.0,  1.0]])

    assert ke.shape == (2, 2)
    assert np.allclose(ke, ke.T)
    assert np.allclose(ke, ke_exact, rtol=1e-8, atol=1e-10)


def test_t1d1_element_internal_force_linear_bar():
    L = 1.0
    A = 1.0
    E = 100.0
    x_e = np.array([0.0, L])

    # u1 = 0, u2 = 0.2 → strain = 0.2, σ = 20
    u_e = np.array([0.0, 0.2])

    f_int = t1d1_element_internal_force(x_e, u_e, A, _test_material(E), ngauss=2)

    assert f_int.shape == (2,)
    assert f_int[0] == pytest.approx(-20.0, rel=1e-8)
    assert f_int[1] == pytest.approx(20.0, rel=1e-8)


def test_t1d1_element_uniform_load_matches_classic_result():
    L = 1.0
    A = 1.0
    x_e = np.array([0.0, L])
    q = 10.0
    direction = 1.0

    f_ext = t1d1_element_uniform_load(x_e, A, q, direction, ngauss=2)

    # classic result: qL/2 at each node
    assert f_ext.shape == (2,)
    assert f_ext[0] == pytest.approx(q * L / 2.0, rel=1e-8)
    assert f_ext[1] == pytest.approx(q * L / 2.0, rel=1e-8)
