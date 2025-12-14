import io

import numpy as np

import wundy
import wundy.first


def test_first_1():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    assert np.allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(F, [0, 0, 0, 0, 2])
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )


def test_first_2():
    """Bar loaded by a uniform axial distributed load (BX).

    The model is the same 4-element bar as in ``test_first_1``:

    * Nodes at x = 0, 1, 2, 3, 4
    * Element connectivity: [1–2], [2–3], [3–4], [4–5]
    * Left end (node 1) fixed in x
    * Uniform line load q = 2.0 applied to all elements in the +x direction

    The expected stiffness matrix is unchanged from ``test_first_1``, while the
    equivalent nodal force vector and displacements are different.  The exact
    reference values below were obtained by solving the same assembled system
    analytically.
    """
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: dload-1
    type: BX
    elements: [1, 2, 3, 4]
    value: 2.0
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: block-1
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]

    # Displacements resulting from the uniform line load q = 2.0
    assert np.allclose(dofs, [0.0, 0.7, 1.2, 1.5, 1.6])

    # Equivalent nodal forces from the distributed load
    assert np.allclose(F, [1.0, 2.0, 2.0, 2.0, 1.0])

    # Stiffness matrix is the same as in the point-load case
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )


def test_gauss_element_stiffness_t1d1():
    """element_stiffness_t1d1 should match the analytic bar stiffness matrix.

    For a 2-node bar of length h = 1, area A = 2, and E = 10, the exact
    stiffness matrix is

        k = (E*A/h) * [[1, -1],
                       [-1, 1]].

    The Gauss-quadrature implementation in element_stiffness_t1d1 should
    reproduce this exactly.
    """
    # Element from x = 0 to x = 1
    xe = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    area = 2.0
    E = 10.0
    h = 1.0

    ke = wundy.first.element_stiffness_t1d1(xe, area, E)

    ke_exact = (E * area / h) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    assert np.allclose(ke, ke_exact)


def test_gauss_element_internal_force_t1d1():
    """element_internal_force_t1d1 should reproduce the exact internal forces.

    For a bar of length h = 1, A = 2, E = 10 with nodal displacements
    u = [0.0, 0.1], the strain is 0.1 and the stress is 1.0. The exact
    internal nodal forces are [-2.0, 2.0].
    """
    xe = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    area = 2.0
    material = {
        "parameters": {"E": 10.0, "nu": 0.3},
    }
    ue = np.array([0.0, 0.1], dtype=float)

    fint = wundy.first.element_internal_force_t1d1(xe, area, material, ue)

    fint_exact = np.array([-2.0, 2.0], dtype=float)
    assert np.allclose(fint, fint_exact)


def test_gauss_points_1d_two_point():
    """Check that the 2-point Gauss rule is implemented correctly."""
    xi, w = wundy.first.gauss_points_1d_two_point()

    # Points should be ±1/sqrt(3), weights should be 1.
    expected_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    expected_w = np.ones(2, dtype=float)

    assert np.allclose(xi, expected_xi)
    assert np.allclose(w, expected_w)


def test_neo_hooke_linearization():
    """Neo-Hooke class model: P(F=1)=0 and tangent equals E."""

    material = {
        "type": "elastic",
        "name": "mat-1",
        "parameters": {"E": 10.0, "nu": 0.3},
    }

    F0 = 1.0
    P0, dP_dF0 = wundy.first.material_neo_hooke_1d_pk1(material, F0)

    # Stress must vanish at the reference state
    assert np.isclose(P0, 0.0, atol=1e-12)

    # Tangent at F=1 should equal E for this 1D Neo-Hooke law
    assert np.isclose(dP_dF0, material["parameters"]["E"], rtol=1e-12)


def test_neo_hooke_finite_strain_differs_from_linear():
    """At finite strain, Neo-Hooke response differs from linear elasticity."""

    material = {
        "type": "elastic",
        "name": "mat-1",
        "parameters": {"E": 10.0, "nu": 0.3},
    }
    E = material["parameters"]["E"]

    F = 1.5  # 50% stretch

    # Neo-Hooke class formula: σ = (E/2)(F - 1/F)
    P_neo, _ = wundy.first.material_neo_hooke_1d_pk1(material, F)

    # Linear 1D prediction: σ = E (F - 1)
    P_lin = E * (F - 1.0)

    # They should not coincide at this finite strain
    assert not np.isclose(P_neo, P_lin)

    # Check against the closed-form value for this case:
    # σ = (E/2)(1.5 - 2/3) = 5 * (0.833333...) ≈ 4.166666...
    P_ref = 4.166666666666667
    assert np.isclose(P_neo, P_ref, rtol=1e-12)


def test_element_residual_t1d1_basic():
    f_int = np.array([1.0, -1.0])
    f_ext = np.array([0.25, -0.75])
    r = wundy.first.element_residual_t1d1(f_int, f_ext)
    assert np.allclose(r, [0.75, -0.25])


def _build_small_load_bar():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: dload-1
    type: BX
    elements: [1, 2, 3, 4]
    value: 0.2          # smaller q than in test_first_2
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: block-1
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    return wundy.ui.preprocess(data)


def test_newton_bar_neo_hooke_matches_linear_for_small_load():
    """For modest loads, Neo-Hooke should be close to linear elasticity.

    The separate test `test_neo_hooke_linearization` already checks that the
    tangent at F = 1 equals E. Here we verify that, for the small load used
    in `_build_small_load_bar`, the nonlinear Neo-Hooke solution produces
    displacements that are close to those from the linear elastic solver.
    """
    inp = _build_small_load_bar()

    # Linear elastic solution
    sol_lin = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    # Nonlinear Neo-Hooke solution
    sol_nl = wundy.first.newton_bar_neo_hooke_1d(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    # Neo-Hooke should be close to linear for modest loads,
    # but some nonlinear deviation is expected.
    assert np.allclose(
        sol_nl["dofs"],
        sol_lin["dofs"],
        rtol=5e-2,  # allow ~5% relative difference
        atol=1e-6,
    )


def test_euler_beam_stiffness_matches_closed_form():
    """Euler beam stiffness should match the standard 4x4 formula."""
    E = 10.0
    I = 2.0
    L = 3.0

    ke = wundy.first.element_stiffness_euler_bernoulli(E, I, L)

    factor = E * I / (L**3)
    ke_exact = factor * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
        ],
        dtype=float,
    )

    assert np.allclose(ke, ke_exact)


def test_euler_beam_uniform_load_vector():
    """Check consistent nodal loads for a uniform transverse load."""
    q = 1.5
    L = 2.0

    fe = wundy.first.element_load_euler_bernoulli_uniform(q, L)

    fe_exact = np.array(
        [
            q * L / 2.0,
            q * L * L / 12.0,
            q * L / 2.0,
            -q * L * L / 12.0,
        ],
        dtype=float,
    )

    assert np.allclose(fe, fe_exact)

def test_arbitrary_dload():
    """element_external_force_t1d1_arbitrary integrates q(x) correctly.

    We take a single element from x=0 to x=1, shape functions

        N1(x) = 1 - x
        N2(x) = x

    and a polynomial distributed load q(x) = x^2. The consistent nodal forces are

        f1 = ∫_0^1 (1 - x) x^2 dx = ∫_0^1 (x^2 - x^3) dx
           = [x^3/3 - x^4/4]_0^1 = 1/3 - 1/4 = 1/12

        f2 = ∫_0^1 x * x^2 dx = ∫_0^1 x^3 dx
           = [x^4/4]_0^1 = 1/4.

    The 2-point Gauss rule should integrate this cubic exactly, so the
    numerical result from element_external_force_t1d1_arbitrary should match
    [1/12, 1/4].
    """

    def q_func(x: float) -> float:
        return x**2

    # Element from x = 0 to x = 1
    xe = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    f_num = wundy.first.element_external_force_t1d1_arbitrary(xe, q_func)

    f_exact = np.array([1.0 / 12.0, 1.0 / 4.0], dtype=float)

    assert np.allclose(f_num, f_exact, rtol=1e-12, atol=1e-12)


def test_equation_profile_bx_via_yaml():
    """BX distributed load with EQUATION profile should assemble correctly.

    We use a single element from x = 0 to x = 1, with q(x) = x^2 in the
    +x direction. The consistent nodal forces are

        f1 = ∫_0^1 (1 - x) x^2 dx = 1/12
        f2 = ∫_0^1 x * x^2 dx     = 1/4.

    The YAML path (load → preprocess → first_fe_code) should reproduce these.
    """
    file = io.StringIO()
    file.write(
        """\
wundy:
  nodes: [[1, 0.0], [2, 1.0]]
  elements: [[1, 1, 2]]
  boundary conditions:
  - name: fix-left
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: dload-eq
    type: BX
    elements: [1]
    profile: EQUATION
    expression: "x**2"
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: block-1
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
"""
    )
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    F = soln["force"]
    assert np.allclose(F, [1.0 / 12.0, 1.0 / 4.0], rtol=1e-12, atol=1e-12)


def _exact_cantilever_uniform_q(
    x: float, L: float, E: float, I: float, q: float
) -> tuple[float, float]:
    """Exact w(x), theta(x) for a cantilever under uniform load q."""
    w = q * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24.0 * E * I)
    theta = q * x * (3 * L**2 - 3 * L * x + x**2) / (6.0 * E * I)
    return w, theta


def test_beam_fe_code_single_element_point_load():
    """Global Euler–Bernoulli solver: single element with end point load.

    We consider one beam element from x=0 to x=L with

        * nodes: [1, 2]
        * E = 10, I = 2, L = 3
        * node 1 fully clamped: w1 = 0, theta1 = 0
        * node 2 loaded by a downward point force F = 1 at the DOF w2

    The global solution from beam_fe_code should match solving
    K u = F directly with the same boundary conditions.
    """
    file = io.StringIO()
    file.write(
        """\
wundy:
  nodes: [[1, 0.0], [2, 3.0]]
  elements: [[1, 1, 2]]
  boundary conditions:
  - name: clamp-left-w
    nodes: [1]
    dof: x
    value: 0.0
  - name: clamp-left-theta
    nodes: [1]
    dof: y
    value: 0.0
  concentrated loads:
  - name: end-force
    nodes: [2]
    dof: x
    value: 1.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: beam-block
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
        I: 2.0
"""
    )
    file.seek(0)

    # YAML → preprocess → beam solver
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    sol = wundy.first.beam_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    u = sol["dofs"]

    # Build reference solution directly
    E = 10.0
    I = 2.0
    L = 3.0
    ke = wundy.first.element_stiffness_euler_bernoulli(E, I, L)

    # DOF ordering per element: [w1, theta1, w2, theta2]
    # Global system is the same here because there is only one element.
    K = ke.copy()

    F = np.zeros(4, dtype=float)
    F[2] = 1.0  # force at w2

    # Clamp node 1: w1 = 0, theta1 = 0  → DOFs 0 and 1 prescribed
    prescribed_idx = np.array([0, 1], dtype=int)
    prescribed_vals = np.array([0.0, 0.0], dtype=float)
    all_dofs = np.arange(4, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, prescribed_idx)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_idx)]
    Ff = F[free_dofs] - Kfp @ prescribed_vals
    uf = np.linalg.solve(Kff, Ff)

    u_ref = np.zeros(4, dtype=float)
    u_ref[free_dofs] = uf
    u_ref[prescribed_idx] = prescribed_vals

    assert np.allclose(u, u_ref)


def test_beam_fe_code_uniform_qy_load():
    """Global Euler–Bernoulli solver: single element with uniform QY load.

    One beam element from x=0 to x=L with:

        * nodes: [1, 2]
        * E = 10, I = 2, L = 2
        * node 1 clamped: w1 = 0, theta1 = 0
        * uniform transverse load q = 1.5 (type QY)

    The beam_fe_code result should match solving K u = F where
    F is assembled with element_load_euler_bernoulli_uniform.
    """
    file = io.StringIO()
    file.write(
        """\
wundy:
  nodes: [[1, 0.0], [2, 2.0]]
  elements: [[1, 1, 2]]
  boundary conditions:
  - name: clamp-left-w
    nodes: [1]
    dof: x
    value: 0.0
  - name: clamp-left-theta
    nodes: [1]
    dof: y
    value: 0.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: qy-load
    type: QY
    elements: [1]
    value: 1.5
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: beam-block
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
        I: 2.0
"""
    )
    file.seek(0)

    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    sol = wundy.first.beam_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    u = sol["dofs"]
    K = sol["stiff"]
    F = sol["force"]

    # Build reference K and F directly for the same problem
    E = 10.0
    I = 2.0
    L = 2.0
    q = 1.5

    ke = wundy.first.element_stiffness_euler_bernoulli(E, I, L)
    fe = wundy.first.element_load_euler_bernoulli_uniform(q, L)

    K_ref = ke.copy()
    F_ref = fe.copy()

    # Clamp node 1: DOFs 0 and 1 prescribed
    prescribed_idx = np.array([0, 1], dtype=int)
    prescribed_vals = np.array([0.0, 0.0], dtype=float)
    all_dofs = np.arange(4, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, prescribed_idx)

    Kff = K_ref[np.ix_(free_dofs, free_dofs)]
    Kfp = K_ref[np.ix_(free_dofs, prescribed_idx)]
    Ff = F_ref[free_dofs] - Kfp @ prescribed_vals
    uf = np.linalg.solve(Kff, Ff)

    u_ref = np.zeros(4, dtype=float)
    u_ref[free_dofs] = uf
    u_ref[prescribed_idx] = prescribed_vals

    # Check both that the assembled K, F are correct and that u matches u_ref
    assert np.allclose(K, K_ref)
    assert np.allclose(F, F_ref)
    assert np.allclose(u, u_ref)


def test_beam_fe_code_cantilever_uniform_qy():
    """Global beam solver vs analytic cantilever with uniform QY load."""
    file = io.StringIO()
    file.write(
        """\
wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0]]
  elements: [[1, 1, 2], [2, 2, 3]]
  boundary conditions:
  - name: clamp-left-w
    nodes: [1]
    dof: x
    value: 0.0
  - name: clamp-left-theta
    nodes: [1]
    dof: y
    value: 0.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: qy-load
    type: QY
    elements: [1, 2]
    value: 1.5
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: beam-block
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
        I: 2.0
"""
    )
    file.seek(0)

    # YAML → preprocess → global beam solver
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    sol = wundy.first.beam_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = sol["dofs"]

    # Material / geometry from YAML
    E = 10.0
    I = 2.0
    L = 2.0
    q = 1.5

    # Nodal coordinates in x (same as inp["coords"][:, 0])
    x_nodes = np.array([0.0, 1.0, 2.0], dtype=float)

    # DOF ordering: [w1, theta1, w2, theta2, w3, theta3]
    for a, x in enumerate(x_nodes):
        w_exact, theta_exact = _exact_cantilever_uniform_q(x, L, E, I, q)
        w_fe = dofs[2 * a]
        theta_fe = dofs[2 * a + 1]

        # 2 elements is a coarse mesh, so use modest tolerances
        assert np.isclose(w_fe, w_exact, rtol=5e-2, atol=1e-6)
        assert np.isclose(theta_fe, theta_exact, rtol=5e-2, atol=1e-6)


def test_beam_fe_code_mms_discrete_global():
    """Continuous MMS: manufactured w(x) for a beam with QY equation load.

    We choose a smooth manufactured solution

        w_exact(x)     = x^5
        theta_exact(x) = w'(x) = 5 x^4

    For an Euler–Bernoulli beam, the governing equation is

        E I w''''(x) = q(x).

    Differentiating w_exact gives w''''(x) = 120 x, so

        q(x) = 120 E I x.

    We then:

      1. Encode q(x) as a QY distributed load with profile = EQUATION
         and expression = "coef_q * x".
      2. Apply Dirichlet boundary conditions at both ends so that
         w and theta match w_exact and theta_exact at x = 0 and x = L.
      3. Solve with beam_fe_code.
      4. Check that the FE nodal DOFs reproduce the manufactured
         values at each node (within a modest tolerance).
    """
    # Material and geometry
    E = 10.0
    I = 2.0
    L = 2.0  # beam from x=0 to x=2 with nodes at 0, 1, 2

    def w_exact(x: float) -> float:
        return x**5

    def theta_exact(x: float) -> float:
        return 5.0 * x**4

    # From w''''(x) = 120 x  ->  q(x) = 120 * E * I * x
    coef_q = 120.0 * E * I

    file = io.StringIO()
    file.write(
        f"""\
wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0]]
  elements: [[1, 1, 2], [2, 2, 3]]
  boundary conditions:
  - name: left-w
    nodes: [1]
    dof: x
    value: {w_exact(0.0)}
  - name: left-theta
    nodes: [1]
    dof: y
    value: {theta_exact(0.0)}
  - name: right-w
    nodes: [3]
    dof: x
    value: {w_exact(L)}
  - name: right-theta
    nodes: [3]
    dof: y
    value: {theta_exact(L)}
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: {E}
      nu: 0.3
  distributed loads:
  - name: qy-mms
    type: QY
    elements: [1, 2]
    profile: EQUATION
    expression: "{coef_q}*x"
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: beam-block
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
        I: {I}
"""
    )
    file.seek(0)

    # YAML → preprocess → global beam solver
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    sol = wundy.first.beam_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = sol["dofs"]

    # Nodal x-coordinates from preprocessed input
    x_nodes = inp["coords"][:, 0]

    # DOF ordering for beam_fe_code is [w1, theta1, w2, theta2, w3, theta3]
    for a, x in enumerate(x_nodes):
        w_fe = dofs[2 * a]
        theta_fe = dofs[2 * a + 1]

        assert np.isclose(w_fe, w_exact(float(x)), rtol=5e-2, atol=1e-8)
        assert np.isclose(theta_fe, theta_exact(float(x)), rtol=5e-2, atol=1e-8)


def test_newton_bar_neo_hooke_single_element_matches_closed_form_finite_strain():
    """
    Nonlinear solver test (not just material law):

    Single 2-node bar, left end fixed, right end has axial point load T.
    For a 1D Neo-Hooke "class model" with P(F) = (E/2)(F - 1/F),
    uniform equilibrium implies T = A*P(F) and F is constant.

    Closed form:
        alpha = 2T/(E A)
        F = (alpha + sqrt(alpha^2 + 4))/2
        u2 = (F - 1)*L
    """
    # Pick values that give clearly non-small strain
    E = 10.0
    A = 1.0
    L = 1.0
    T = 7.5  # this produces F=2 exactly for this model when E=10, A=1

    file = io.StringIO()
    file.write(
        f"""\
wundy:
  nodes: [[1, 0.0], [2, {L}]]
  elements: [[1, 1, 2]]
  boundary conditions:
  - name: fix-left
    dof: x
    nodes: [1]
    value: 0.0
  concentrated loads:
  - name: end-force
    nodes: [2]
    value: {T}
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: {E}
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: ALL
    element:
      type: t1d1
      properties:
        area: {A}
"""
    )
    file.seek(0)

    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)

    sol = wundy.first.newton_bar_neo_hooke_1d(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
        max_iter=50,
        tol=1e-12,
    )

    u = sol["dofs"]

    # --- Closed-form expected displacement ---
    alpha = 2.0 * T / (E * A)
    F_exact = 0.5 * (alpha + np.sqrt(alpha * alpha + 4.0))
    u2_exact = (F_exact - 1.0) * L

    # Left is fixed, right should match analytic result
    assert np.isclose(u[0], 0.0, atol=1e-14)
    assert np.isclose(u[1], u2_exact, rtol=1e-12, atol=1e-12)

    # Extra solver-level sanity checks (optional but nice):
    # 1) The computed stretch from FE kinematics matches F_exact
    F_fe = 1.0 + (u[1] - u[0]) / L
    assert np.isclose(F_fe, F_exact, rtol=1e-12, atol=1e-12)

    # 2) Force balance: internal - external residual on free DOF is ~0
    #    (Node 2 is the only free DOF here)
    R = sol["residual"]
    assert np.isclose(R[1], 0.0, atol=1e-10)
