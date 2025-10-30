import io

import numpy as np

from wundy import ui
from wundy.wundy import solve


def test_second_1():
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
    elements: [1, 2]
    element:
      type: T1D2
      properties:
        area: 1
  - material: mat-1
    name: block-2
    elements: [3, 4]
    element:
      type: T1D2
      properties:
        area: 1
""")
    file.seek(0)
    data = ui.load(file)
    inp = ui.preprocess(data)
    soln = solve(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["equations"],
        inp["block_elem_map"],
        inp["solver"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    assert np.allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8])
    R = np.dot(K, dofs) - F
    assert R[0] == -2.0
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
