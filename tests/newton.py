import numpy as np
import wundy
from wundy.schemas import DIRICHLET
from wundy.schemas import NEUMANN

def test_newton_solve_LE():
    coords = np.array([[0.0], [1.0]])
    blocks = [{
        "connect": [[0, 1]],
        "material": "mat1",
        "element": {"properties": {"area": 1}}
    }]
    materials = {
        "mat1": {
            "type": "NEOHOOK",
            "parameters": {"mu": 1, "lam": 0}
        }
    }
    bcs = [
        {"type": DIRICHLET, "nodes": [0], "dof": 0, "value": 0.0},
        {"type": NEUMANN, "nodes": [1], "dof": 0, "value": 1.0}
    ]
    dloads = []
    block_elem_map = {0: (0, 0)}

    solution = wundy.first.first_fe_code(
        coords,
        blocks,
        bcs,
        dloads,
        materials,
        block_elem_map
    )

    u_expected = 1.0
    u_computed = solution["dofs"][1]

    assert solution["converged"] is True, "Newton Solver failed to converge for linear material"
    assert solution["num_iter"] == 1, "Newton Solver exceeded 1 iteration for linear material"
    assert np.isclose(u_computed, u_expected, atol = 1e-8), f"Computed displacement {u_computed} does not match expected {u_expected}"
    return None

def test_newton_solve_neo_small_strain():
    coords = np.array([[0.0], [1.0]])
    blocks = [{
        "connect": [[0, 1]],
        "material": "mat1",
        "element": {"properties": {"area": 1}}
    }]
    materials = {
        "mat1": {
            "type": "NEOHOOK",
            "parameters": {"mu": 1, "lam": 0}
        }
    }
    bcs = [
        {"type": DIRICHLET, "nodes": [0], "dof": 0, "value": 0.0},
        {"type": NEUMANN, "nodes": [1], "dof": 0, "value": 1.0}
    ]
    dloads = []
    block_elem_map = {0: (0, 0)}

    solution = wundy.first.first_fe_code(
        coords,
        blocks,
        bcs,
        dloads,
        materials,
        block_elem_map
    )

    u_expected = 1.0
    u_computed = solution["dofs"][1]

    assert solution["converged"] is True, "Newton Solver failed to converge for Neo-Hookean material"
    assert solution["num_iter"] <= 5, "Newton Solver exceeded expected iterations for smalls strain Neo-Hookean material"
    assert np.isclose(u_computed, u_expected, atol = 1e-6), f"Computed displacement {u_computed} does not match expected {u_expected}"
    return None