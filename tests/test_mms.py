import numpy as np
import yaml
import io

import wundy


# -------------------------------------------------------------
# Manufactured exact solution
# -------------------------------------------------------------
def manufactured_solution(x):
    return np.sin(np.pi * x)


def manufactured_force(x):
    return np.pi**2 * np.sin(np.pi * x)


# -------------------------------------------------------------
# Build MMS YAML input
# -------------------------------------------------------------
def build_mms_yaml(nnodes=40, L=1.0, E=1.0, A=1.0):
    xs = np.linspace(0, L, nnodes)
    nodes_yaml = "\n".join(f"    - [{i+1}, {x}]" for i, x in enumerate(xs))

    elements_yaml = "\n".join(
        f"    - [{i+1}, {i+1}, {i+2}]" for i in range(nnodes - 1)
    )

    # MMS forcing: FE requires Fext = -f(x)
    fvals = -manufactured_force(xs)

    loads_yaml_list = []
    for i in range(nnodes):
        loads_yaml_list.append(
            f"    - nodes: [{i+1}]\n"
            f"      dof: x\n"
            f"      value: {fvals[i]}"
        )
    loads_yaml = "\n".join(loads_yaml_list)

    text = f"""
wundy:
  nodes:
{nodes_yaml}

  elements:
{elements_yaml}

  boundary conditions:
    - nodes: [1]
      dof: x
      type: dirichlet
      value: 0.0
    - nodes: [{nnodes}]
      dof: x
      type: dirichlet
      value: 0.0

  materials:
    - type: elastic
      name: MAT
      parameters:
        E: {E}
        nu: 0.3

  element blocks:
    - name: BLOCK
      material: MAT
      elements: all
      element:
        type: T1D1
        properties:
          area: {A}

  concentrated loads:
{loads_yaml}
"""
    return text


# -------------------------------------------------------------
# Helper to run
# -------------------------------------------------------------
def run_mms(nnodes=40):
    yaml_text = build_mms_yaml(nnodes)
    stream = io.StringIO(yaml_text)
    data = wundy.ui.load(stream)
    pre = wundy.ui.preprocess(data)

    result = wundy.solver.newton_solve_1d(pre)
    u = result["dofs"]
    return u, pre


# -------------------------------------------------------------
# Actual test
# -------------------------------------------------------------
def test_mms_solution_accuracy():
    nnodes = 40
    u, pre = run_mms(nnodes)

    xs = pre["coords"][:, 0]
    u_exact = manufactured_solution(xs)

    assert np.allclose(u, u_exact, atol=1e-2)
