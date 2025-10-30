import io

import numpy as np

import wundy.model


def test_model_1():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes:
  - [1, 0, 0]
  - [2, 1, 0]
  - [3, 2, 0]
  - [4, 0, 1]
  - [5, 1, 1]
  - [6, 2, 1]
  - [7, 0, 2]
  - [8, 1, 2]
  - [9, 2, 2]
  elements:
  #   7--(5)--8--(6)--9
  #   |       |       |
  # (10)    (11)   (12)
  #   |       |       |
  #   4--(3)--5--(4)--6
  #   |       |       |
  #  (7)     (8)     (9)
  #   |       |       |
  #   1--(1)--2--(2)--3
  - [1, 1, 2]
  - [2, 2, 3]
  - [3, 4, 5]
  - [4, 5, 6]
  - [5, 7, 8]
  - [6, 8, 9]
  - [7, 1, 4]
  - [8, 2, 5]
  - [9, 3, 6]
  - [10, 4, 7]
  - [11, 5, 8]
  - [12, 6, 9]
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
    elements: [1, 2, 3, 10]
    element:
      type: T1D2
      properties:
        area: 1
  - material: mat-1
    name: block-2
    elements: [4, 5, 6, 11]
    element:
      type: T2D2
      properties:
        area: 1
  - material: mat-1
    name: block-3
    elements: [7, 8, 9, 12]
    element:
      type: B1D2
      properties:
        I: 1
""")
    file.seek(0)
    model = wundy.model.Model.from_file(file)
    model.prepare()
    assert np.allclose(model.dof_map[0, [0, 1, 3]], [0, 1, 2])
    assert np.allclose(model.dof_map[1, [0, 1, 3]], [3, 4, 5])
    assert np.allclose(model.dof_map[2, [0, 1, 3]], [6, 7, 8])
    assert np.allclose(model.dof_map[3, [0, 1, 3]], [9, 10, 11])
    assert np.allclose(model.dof_map[4, [0, 1, 3]], [12, 13, 14])
    assert np.allclose(model.dof_map[5, [0, 1, 3]], [15, 16, 17])
    assert np.allclose(model.dof_map[6, [0, 1]], [18, 19])
    assert np.allclose(model.dof_map[7, [0, 1]], [20, 21])
    assert np.allclose(model.dof_map[8, [0, 1]], [22, 23])
    assert np.allclose(model.block_ele_map[0], [0, 1, 2, 9])
    assert np.allclose(model.block_ele_map[1], [3, 4, 5, 10])
    assert np.allclose(model.block_ele_map[2], [6, 7, 8, 11])
    assert np.allclose(model.block_nod_map[0], [0, 1, 2, 3, 4, 6, -1])
    assert np.allclose(model.block_nod_map[1], [4, 5, 6, 7, 8, -1, -1])
    assert np.allclose(model.block_nod_map[2], [0, 1, 2, 3, 4, 5, 8])
    assert np.allclose(model.block_dof_map[0], [0, 3, 6, 9, 12, 18, -1, -1, -1, -1, -1, -1, -1, -1])
    assert np.allclose(
        model.block_dof_map[1], [12, 13, 15, 16, 18, 19, 20, 21, 22, 23, -1, -1, -1, -1]
    )
    assert np.allclose(model.block_dof_map[2], [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 22, 23])
