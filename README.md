# wundy
**Wundy** is a finite element (FE) code for
one-dimensional problems. It is written in pure Python and focuses on clarity and testability.

The current version supports:

- Linear, static **axial bar** problems (T1D1 element, axial DOF)
- A 1D **Neo-Hooke** bar solved with Newton’s method
- **Arbitrary axial distributed loads** specified as Python expressions
- **Euler–Bernoulli beam** elements with transverse displacement and rotation
- A small YAML-based input format with schema validation
- Method of Manufactured Solutions (MMS) tests for both bar and beam cases

All core functionality is covered by unit tests in `tests/`.

## Documentation
- [User Manual](docs/user_manual.md)

### Clone repository

```console
git clone git@github.com:Ozan-ALtuntas/wundy
```

## Installation

To install Wundy, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate         # on Linux/macOS
# or
.\.venv\Scripts\Activate.ps1      # on Windows PowerShell

pip install -e .
pip install -r requirements.txt   # if you have one
```
## Testing

In the `wundy` directory, execute

```console
pytest
```
or

```console
ruff format src
ruff check --fix src
ty check src
pytest
```
for more comprehensive testing. All current tests live under tests/ and should pass when
everything is set up correctly.

### Element types

- `T1D1` (2-node 1D element) used in two roles:
  - Axial bar (linear elasticity, Neo-Hooke bar)
  - Euler–Bernoulli beam (bending in a single plane)

### Materials

- `ELASTIC`  
  - Parameters: `E`, `nu`  
  - Used for both bar and beam problems.
- Neo-Hooke 1D bar:
  - Implemented via `material_neo_hooke_1d_pk1` and the global
    solver `newton_bar_neo_hooke_1d`.
  - At small strain, the tangent matches `E` (verified by tests).

### Loads

- **Concentrated (nodal) loads**:
  - Applied via `concentrated loads` in the YAML.
  - Treated as **Neumann** conditions in the global assembly.

- **Distributed loads** (`distributed loads` in YAML):
  - `type: BX`  
    Axial line load along the bar / beam axis.
    - `profile: UNIFORM` with `value: q` (constant line load)
    - `profile: EQUATION` with `expression: "x**2"` etc.  
      Uses Gauss integration of a user-supplied Python expression `q(x)`.
  - `type: GRAV`  
    Body force in the axial direction (treated similarly to uniform `BX`).
  - `type: QY`  
    Uniform transverse load on Euler–Bernoulli beam elements.  
    Uses the standard 4×1 consistent nodal load vector for a uniform `q`.

---

## Repository Layout

- `src/wundy/`
  - `__init__.py`
  - `schemas.py` – input validation, type normalization, default insertion
  - `ui.py`      – YAML loader and preprocessing to solver-ready data
  - `first.py`   – element routines and global solvers
- `tests/`
  - `first.py`       – unit tests for elements, global bar/beam solvers,
                       distributed loads, Neo-Hooke, and MMS cases
  - `user_input.py`  – tests for YAML validation and preprocessing
- `pyproject.toml`   – tooling configuration (`ruff`, `pytest`, `typing`)

---

## Quickstart Example – Axial Bar with Point Load

Minimal YAML input ("bar_example.yml"):

```yaml:
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]

  boundary conditions:
  - name: fix-left
    dof: x
    nodes: [1]

  concentrated loads:
  - name: cload-1
    nodes: [5]
    dof: x
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
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1.0
```

In Python driver:

```python:
import io
import wundy

file = io.StringIO("bar_example.yml")
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

print("Displacements:", soln["dofs"])
print("Stiffness matrix:\n", soln["stiff"])
print("Force vector:", soln["force"])
```

---

## Quickstart Example – Euler–Bernoulli Beam with Uniform QY Load

YAML for a single beam element from 0 to L=2 fixed on the left and uniform downward load q=1.5 ("beam_example.yml"):

```yaml:
wundy:
  nodes: [[1, 0.0], [2, 2.0]]
  elements: [[1, 1, 2]]

  boundary conditions:
  - name: clamp-left-w
    nodes: [1]
    dof: x      # transverse displacement w1 = 0
    value: 0.0
  - name: clamp-left-theta
    nodes: [1]
    dof: y      # rotation theta1 = 0
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
```

In Python driver:

```python:
import io
import wundy

file = io.StringIO(open("beam_example.yml").read())
data = wundy.ui.load(file)
inp = wundy.ui.preprocess(data)

soln = wundy.first.beam_fe_code(
    inp["coords"],
    inp["blocks"],
    inp["bcs"],
    inp["dload"],
    inp["materials"],
    inp["block_elem_map"],
)

print("Beam DOFs (w1, theta1, w2, theta2, ...):", soln["dofs"])
```

---

## Limitations

1D geometry only (bars and beams).

Small-deformation kinematics (except for the 1D Neo-Hooke bar, which
is still a simple model).

Only one element type T1D1, reused for bar and beam.

No GUI or full CLI; the primary interface is Python + YAML input.
