import io

import numpy as np

import wundy
import wundy.first
def _run(yaml_text: str):
    import io
    data = wundy.ui.load(io.StringIO(yaml_text))
    inp = wundy.ui.preprocess(data)
    # New schema keys from preprocess:
    #   coords, blocks, bcs, dload, materials, block_elem_map
    return wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )




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
    """
    Uniform distributed load represented via equivalent nodal forces (cloads),
    new schema (nodes / elements / boundary conditions ...).

    Bar: x = [0,1,2,3,4], 4 unit elements, EA = 10, left end fixed.
    Equivalent nodal forces for q=1 per unit element:
      F ≈ [0.0, 1.0, 1.0, 1.0, 0.5]
    Expected displacements (approx):
      u ≈ [0.00, 0.35, 0.60, 0.75, 0.80]
    """
    yaml_text = """
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
    - name: fix-nodes
      dof: x
      nodes: [1]
  concentrated loads:
    - name: cload-q1
      nodes: [2, 3, 4]
      value: 1.0
    - name: cload-tip
      nodes: [5]
      value: 0.5
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
"""
    soln = _run(yaml_text)

    import numpy as np
    # Accept any of the possible displacement keys used across versions
    if "displ" in soln:
        u = np.asarray(soln["displ"])
    elif "u" in soln:
        u = np.asarray(soln["u"])
    elif "U" in soln:
        u = np.asarray(soln["U"])
    elif "dofs" in soln:           # updated return key in newer code
        u = np.asarray(soln["dofs"]).reshape(-1)
    else:
        raise KeyError(f"No displacement key found. Keys present: {list(soln.keys())}")

    u_exp = np.array([0.00, 0.35, 0.60, 0.75, 0.80])
    assert u.shape == (5,)
    assert np.allclose(u, u_exp, rtol=1e-3, atol=1e-6)