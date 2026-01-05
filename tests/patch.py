import io

import numpy as np

from wundy.model import Model


def test_patch_bar_4():
    L = 1.0
    nodes = [[1, 0.0], [2, 0.2], [3, 0.5], [4, 0.7], [5, L]]
    elements = [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
    model = Model(nodes, elements)
    E = 210e9
    A = 1.0
    model.add_material("steel", "elastic", {"E": E})
    model.add_element_block(
        "Block-1",
        "steel",
        {"type": "T1D2", "properties": {"area": A}},
        [1, 2, 3, 4],
    )
    ubar = 0.001
    model.add_dirichlet_boundary("Dirichlet-1", [1], 1, 0.0)
    model.add_dirichlet_boundary("Dirichlet-2", [5], 1, ubar)
    model.solve()
    u = model.solution["dofs"]
    K = model.solution["stiff"]
    F = model.solution["force"]
    xc = np.linspace(0.0, L, model.coords.shape[0])
    uc = np.linspace(0.0, ubar, model.coords.shape[0])
    u_exact = np.interp(model.coords.flatten(), xc, uc)
    strain_exact = (ubar - 0.0) / L
    stress_exact = E * strain_exact
    reaction_exact = -stress_exact * A
    disp_err = np.linalg.norm(model.solution["dofs"] - u_exact)
    assert disp_err < 1e-12, f"Patch test failed: displacement error {disp_err}"
    reaction = np.dot(K, u) - F
    assert np.allclose(reaction[0], reaction_exact, rtol=1e-12)


def test_patch_bar_dload():
    L = 1.0
    nodes = [[1, 0.0], [2, 0.25], [3, 0.5], [4, 0.75], [5, L]]
    elements = [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
    model = Model(nodes, elements)
    A = 1.0
    E = 210e9
    model.add_material("steel", "elastic", {"E": E})
    model.add_element_block(
        "Block-1",
        "steel",
        {"type": "T1D2", "properties": {"area": A}},
        [1, 2, 3, 4],
    )
    model.add_dirichlet_boundary("Dirichlet-1", [1], 1, 0.0)
    q = 1000.0
    model.add_distributed_load("DLoad-1", "BX", [1, 2, 3, 4], q, [1.0])
    model.solve()
    u = model.solution["dofs"]
    K = model.solution["stiff"]
    F = model.solution["force"]
    # Analytic solution
    # u(x) = (q * x**2) / 2 / E / A
    u_exact = np.zeros_like(u)
    for i, x in enumerate(model.coords[:, 0]):
        u_exact[i] = (q * x**2) / (2.0 * E * A) * (L - x)
    disp_err = np.linalg.norm(u - u_exact)
    assert disp_err < 1e-8, f"Patch test failed: displacement error {disp_err}"
    reaction_exact = -q * L
    reaction = np.dot(K, u) - F
    assert np.allclose(reaction[0], reaction_exact, rtol=1e-8)


def test_patch_mpc():
    A = 1.0
    E = 1.0
    nodes = [[1, 0.0], [2, 1.0], [3, 2.0], [4, 3.0], [5, 4.0], [6, 5.0]]
    elements = [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 6]]
    model = Model(nodes, elements)
    model.add_material("Material-1", "elastic", {"E": E})
    model.add_element_block(
        "Block-1",
        "Material-1",
        {"type": "T1D2", "properties": {"area": A}},
        [1, 2, 3, 4, 5],
    )
    model.add_dirichlet_boundary("Dirichlet-1", [1], 1, 0.0)
    model.add_distributed_load("DLoad-1", "BX", [1, 2, 3, 4, 5], 10.0, [1.0])
    model.add_equation([(2, 1, 1.0), (5, 1, -1.0)])
    model.add_equation([(3, 1, 1.0), (6, 1, -1.0)])
    model.solve()
    u = model.solution["dofs"]
    assert np.allclose(u[1], u[4], rtol=1e-12)
    assert np.allclose(u[2], u[5], rtol=1e-12)


def test_patch_mpc_inp():
    f = io.StringIO()
    f.write("""\
wundy:
  nodes:
  - [1, 0.0]
  - [2, 1.0]
  - [3, 2.0]
  - [4, 3.0]
  - [5, 4.0]
  - [6, 5.0]
  elements:
  - [1, 1, 2]
  - [2, 2, 3]
  - [3, 3, 4]
  - [4, 4, 5]
  - [5, 5, 6]
  materials:
  - name: mat-1
    type: elastic
    parameters:
      E: 1.0
      nu: 0.0
  element blocks:
  - name: block-1
    element:
      type: T1D2
      properties:
        area: 1.0
    elements: elset-1
    material: mat-1
  boundary conditions:
  - name: bc-1
    type: dirichlet
    nodes: [1]
    dof: x
    value: 0.0
  element sets:
  - name: elset-1
    elements: [1, 2, 3, 4, 5]
  distributed loads:
  - name: dload-1
    type: BX
    direction: [1.0]
    elements: elset-1
    value: 10.0
  equations:
  - u[2, 1] - 2 * u[5, 1]
  - u[3, 1] - 3 * u[6, 1]
""")
    f.seek(0)
    model = Model.from_file(f)
    model.solve()
    u = model.solution["dofs"]
    assert np.allclose(u[1], 2 * u[4], rtol=1e-12)
    assert np.allclose(u[2], 3 * u[5], rtol=1e-12)


def test_patch_mpc_dummy():
    f = io.StringIO()
    f.write("""\
wundy:
  nodes:
  - [1, 0.0]
  - [2, 1.0]
  - [3, 2.0]
  - [4, 3.0]
  - [5, 4.0]
  - [6, 5.0]
  - [100, 0.0]
  elements:
  - [1, 1, 2]
  - [2, 2, 3]
  - [3, 3, 4]
  - [4, 4, 5]
  - [5, 5, 6]
  materials:
  - name: mat-1
    type: elastic
    parameters:
      E: 1.0
      nu: 0.0
  element blocks:
  - name: block-1
    element:
      type: T1D2
      properties:
        area: 1.0
    elements: elset-1
    material: mat-1
  boundary conditions:
  - name: bc-1
    type: dirichlet
    nodes: [1]
    dof: x
    value: 0.0
  - name: bc-2
    type: dirichlet
    nodes: [100]
    dof: x
    value: 3.4
  element sets:
  - name: elset-1
    elements: [1, 2, 3, 4, 5]
  equations:
  - u[6, 1] - u[100, 1]
""")
    f.seek(0)
    model = Model.from_file(f)
    model.solve()
    u = model.solution["dofs"]
    assert np.allclose(u[5], 3.4, rtol=1e-12)


def test_patch_bar_4_newton():
    L = 1.0
    nodes = [[1, 0.0], [2, 0.2], [3, 0.5], [4, 0.7], [5, L]]
    elements = [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
    A = 1.0
    E = 210.0e9
    model = Model(nodes, elements)
    model.add_material("steel", "elastic", {"E": E})
    model.add_element_block(
        "Block-1",
        "steel",
        {"type": "T1D2", "properties": {"area": A}},
        [1, 2, 3, 4],
    )
    ubar = 0.001
    model.add_dirichlet_boundary("Dirichlet-1", [1], 1, 0.0)
    model.add_dirichlet_boundary("Dirichlet-2", [5], 1, ubar)
    model.set_solver("NONLINEAR", "NEWTON")
    model.solve()
    u = model.solution["dofs"]
    K = model.solution["stiff"]
    F = model.solution["force"]
    xc = np.linspace(0.0, L, model.coords.shape[0])
    uc = np.linspace(0.0, ubar, model.coords.shape[0])
    u_exact = np.interp(model.coords.flatten(), xc, uc)
    strain_exact = (ubar - 0.0) / L
    stress_exact = E * strain_exact
    reaction_exact = -stress_exact * A
    disp_err = np.linalg.norm(model.solution["dofs"] - u_exact)
    assert disp_err < 1e-12, f"Patch test failed: displacement error {disp_err}"
    with np.printoptions(precision=4):
        print("u", u)
        print("K.u", np.dot(K, u))
        print("F", F)
        reaction = np.dot(K, u) - F
        print("R", reaction)
        print(reaction_exact)
    assert np.allclose(reaction[0], reaction_exact, rtol=1e-12)


def test_patch_mpc_dummy_nonlinear():
    f = io.StringIO()
    f.write("""\
wundy:
  nodes:
  - [1, 0.0]
  - [2, 1.0]
  - [3, 2.0]
  - [4, 3.0]
  - [5, 4.0]
  - [6, 5.0]
  - [100, 0.0]
  elements:
  - [1, 1, 2]
  - [2, 2, 3]
  - [3, 3, 4]
  - [4, 4, 5]
  - [5, 5, 6]
  materials:
  - name: mat-1
    type: elastic
    parameters:
      E: 1.0
      nu: 0.0
  element blocks:
  - name: block-1
    element:
      type: T1D2
      properties:
        area: 1.0
    elements: elset-1
    material: mat-1
  boundary conditions:
  - name: bc-1
    type: dirichlet
    nodes: [1]
    dof: x
    value: 0.0
  - name: bc-2
    type: dirichlet
    nodes: [100]
    dof: x
    value: 3.4
  element sets:
  - name: elset-1
    elements: [1, 2, 3, 4, 5]
  solver:
    type: nonlinear
    options:
      method: newton
      max_iterations: 25
      tolerance: .000001
  equations:
  - u[6, 1] - u[100, 1]
""")
    f.seek(0)
    model = Model.from_file(f)
    model.solve()
    u = model.solution["dofs"]
    assert np.allclose(u[5], 3.4, rtol=1e-12)
