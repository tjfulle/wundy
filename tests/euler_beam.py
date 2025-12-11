import numpy as np
import wundy


def test_euler_beam_elem_stiff():
    """The 2D Euler-Bernoulli beam element stiffness should match the closed form."""

    E = 210.0e9  # Young's modulus (Pa)
    I = 8.5e-6  # Area moment of inertia (m^4)
    L = 2.5  # Element length (m)

    xe = np.array([[0.0], [L]])
    material = {"type": "ELASTIC", "parameters": {"E": E}}
    props = {"properties": {"inertia": I, "area": 1.0}}

    ke = wundy.first.elem_stiff(material, xe, props, n_gauss=2)

    factor = E * I / L**3
    expected = factor * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ]
    )

    assert np.allclose(ke, expected)

def test_cantileveer_tip_deflection():
    """A single beam element cantilever should reproduce the classical tip deflection."""

    E = 70.0e9  # Pa
    I = 4.0e-6  # m^4
    L = 1.2  # m
    P = 750.0  # N, downward tip load

    xe = np.array([[0.0], [L]])
    material = {"type": "ELASTIC", "parameters": {"E": E}}
    props = {"properties": {"inertia": I, "area": 1.0}}

    ke = wundy.first.elem_stiff(material, xe, props, n_gauss=2)

    # Fixed degrees of freedom at node 1: transverse displacement and rotation.
    free_dof = slice(2, 4)
    k_ff = ke[free_dof, free_dof]
    f_ext = np.array([P, 0.0])

    u_free = np.linalg.solve(k_ff, f_ext)
    tip_deflection = u_free[0]

    expected = P * L**3 / (3.0 * E * I)
    assert np.isclose(tip_deflection, expected)
