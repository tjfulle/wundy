import io

import numpy as np

import wundy.ui
from wundy.wundy import solve


def test_patch_bar_4():
    L = 1.0
    coords = np.array([[0.0], [0.2], [0.5], [0.7], [L]])
    connect = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    A = 1.0
    nft = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    block = {
        "connect": connect,
        "element": {"properties": {"area": A, "node_freedoms": [nft, nft]}},
        "material": "steel",
    }
    blocks = [block]
    E = 210e9
    steel = {"parameters": {"E": E}}
    materials = {"steel": steel}
    ubar = 0.001
    bcs = [
        {"type": 1, "nodes": [0], "local_dof": 0, "value": 0.0},
        {"type": 1, "nodes": [4], "local_dof": 0, "value": ubar},
    ]
    dloads = []
    block_elem_map = {e: (0, e) for e in range(connect.shape[0])}
    equations = []
    soln = solve(coords, blocks, bcs, dloads, materials, equations, block_elem_map)
    u = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    xc = np.linspace(0.0, L, coords.shape[0])
    uc = np.linspace(0.0, ubar, coords.shape[0])
    u_exact = np.interp(coords.flatten(), xc, uc)
    strain_exact = (ubar - 0.0) / L
    stress_exact = E * strain_exact
    reaction_exact = -stress_exact * A
    disp_err = np.linalg.norm(soln["dofs"] - u_exact)
    assert disp_err < 1e-12, f"Patch test failed: displacement error {disp_err}"
    reaction = np.dot(K, u) - F
    assert np.allclose(reaction[0], reaction_exact, rtol=1e-12)


def test_patch_bar_dload():
    L = 1.0
    coords = np.array([[0.0], [0.25], [0.5], [0.75], [L]])
    connect = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    A = 1.0
    nft = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    block = {
        "connect": connect,
        "element": {"properties": {"area": A, "node_freedoms": [nft, nft]}},
        "material": "steel",
    }
    blocks = [block]
    E = 210e9
    steel = {"parameters": {"E": E}}
    materials = {"steel": steel}
    bcs = [{"type": 1, "nodes": [0], "local_dof": 0, "value": 0.0}]
    q = 1000.0
    dload = {
        "type": "BX",
        "direction": [1.0],
        "elements": [0, 1, 2, 3],
        "value": q,
        "name": "dload-1",
    }
    dloads = [dload]
    block_elem_map = {e: (0, e) for e in range(connect.shape[0])}
    equations = []
    soln = solve(coords, blocks, bcs, dloads, materials, equations, block_elem_map)
    u = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    # Analytic solution
    # u(x) = (q * x**2) / 2 / E / A
    u_exact = np.zeros_like(u)
    for i, x in enumerate(coords[:, 0]):
        u_exact[i] = (q * x**2) / (2.0 * E * A) * (L - x)
    disp_err = np.linalg.norm(u - u_exact)
    assert disp_err < 1e-8, f"Patch test failed: displacement error {disp_err}"
    reaction_exact = -q * L
    reaction = np.dot(K, u) - F
    assert np.allclose(reaction[0], reaction_exact, rtol=1e-8)


def test_patch_mpc():
    L = 1.0
    coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    A = 1.0
    nft = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    connect = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    blocks = [
        {
            "connect": connect,
            "element": {"type": "T1D1", "properties": {"area": A, "node_freedoms": [nft, nft]}},
            "material": "mat1",
        }
    ]
    materials = {"mat1": {"parameters": {"E": 1.0, "nu": 0.0}}}
    bcs = [{"type": 1, "nodes": [0], "local_dof": 0, "value": 0.0}]
    dloads = []
    dload = {
        "type": "BX",
        "direction": [1.0],
        "elements": [0, 1, 2, 3, 4],
        "value": 10.0,
        "name": "dload-1",
    }
    dloads = [dload]
    block_elem_map = {e: (0, e) for e in range(connect.shape[0])}
    equations = [[(1, 0, 1.0), (4, 0, -1.0)], [(2, 0, 1.0), (5, 0, -1.0)]]
    soln = solve(coords, blocks, bcs, dloads, materials, equations, block_elem_map)
    u = soln["dofs"]
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
      type: t1d1
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
    data = wundy.ui.load(f)
    inp = wundy.ui.preprocess(data)
    soln = solve(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["equations"],
        inp["block_elem_map"],
    )
    u = soln["dofs"]
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
      type: t1d1
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
    data = wundy.ui.load(f)
    inp = wundy.ui.preprocess(data)
    soln = solve(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["equations"],
        inp["block_elem_map"],
    )
    u = soln["dofs"]
    assert np.allclose(u[5], 3.4, rtol=1e-12)
