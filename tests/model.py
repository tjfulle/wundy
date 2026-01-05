import io
import pathlib
from typing import Generator

import numpy as np
import pytest

import wundy
import wundy.model


@pytest.fixture
def inp_dir() -> Generator[pathlib.Path, None, None]:
    d = pathlib.Path(wundy.__file__).parent
    root = d.parent.parent
    yield root / "tests/inputs"


def test_model_1(inp_dir):
    file = inp_dir / "mixed-el-1d.yaml"
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


def test_model_2():
    q, L = 8.0, 4.0
    file = io.StringIO()
    file.write(f"""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, {L}]]
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
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: T1D2
      properties:
        area: 1
  distributed loads:
  - name: dload-1
    elements: all
    type: BX
    direction: [1]
    value: {q}
""")
    file.seek(0)
    model = wundy.model.Model.from_file(file)
    model.solve()
    dofs = model.solution["dofs"]
    K = model.solution["stiff"]
    F = model.solution["force"]
    R = np.dot(K, dofs) - F
    assert R[0] == -q * L
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
