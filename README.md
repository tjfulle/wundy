# wundy

wundy is a finite element analysis program for axial bars and Euler-Bernoulli beam. It builds and solves linear elastic problems using YAML-based input that describes the mesh, material properties, loading, and boundary conditions. The tool is designed for lightweight experimentation and validation of 1D bar and truss-style models.

---

## Installation

Install wundy directly from GitHub in a virtual environment:

1. **Clone the repository**

   ```bash
   git clone git@github.com:klw02/wundy
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install wundy in editable mode**

   ```bash
   cd wundy
   python3 -m pip install -e .
   ```

4. **(Optional) Run tests**

   ```bash
   pytest
   ```

## Quick start

1. Activate the environment where you installed wundy.
2. Run a YAML input through the CLI:

   ```bash
   # Using the installed entrypoint
   wundy yaml/simple.yaml

   # Or directly via module (no install, just the source)
   PYTHONPATH=src python -m wundy.run yaml/simple.yaml
   ```

The solver prints a concise summary (converged flag, iterations, DOFs, residual, nodal forces).

## User Input

wundy reads a YAML file with a top-level `wundy` key. Each subsection below defines the expected structure and optional fields.

- **nodes** – Coordinates keyed by ID: `nodes: [[id, x], ...]`.

- **elements** – Element connectivity: `elements: [[element_id, start_node, end_node], ...]`.

- **materials** – Material definitions. Supported `type` values include `elastic` (and `NEOHOOK`, though Neohook validation is not yet implemented). Parameters include Young's modulus `E` (required), Poisson's ratio `nu` (optional, defaults to 0.0), and `density` (optional, defaults to 0.0).

- **element blocks** – Groups elements and assigns material plus element properties. Each block includes a `material`, `elements`, and an `element` section with `type` (e.g., `T1D1`) and `properties` such as `area`.

- **boundary conditions** – Supports `DIRICHLET` (prescribed displacement) and `NEUMANN` (prescribed traction/force). Each condition lists `dof`, `nodes`, and `value`.

- **distributed loads** – Constant loads over elements. Valid `type` values are `BX` for mechanical distributed load and `GRAV` for gravitational body force. Provide `direction`, `value`, and target `elements`.

- **node sets** *(optional)* – Named groups of nodes: `node sets: [{name, nodes}]`.

- **element sets** *(optional)* – Named groups of elements: `element sets: [{name, elements}]`.

- **concentrated loads** *(optional)* – Equivalent to `NEUMANN` boundary conditions but without specifying `type`; include `dof`, `nodes`, and `value`.

## Full Example YAML Inputs

### 2D neohookean beam (Euler–Bernoulli)

```yaml
wundy:
  nodes: [[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 2.0, 0.0]]
  elements: [[1, 1, 2], [2, 2, 3]]

  elements:
    - [1, 1, 2]
    - [2, 2, 3]
  boundary conditions:
  - name: clamp-w
    dof: W
    nodes: [1]
    value: 0.0
  - name: clamp-theta
    dof: THETA
    nodes: [1]
    value: 0.0

  distributed loads:
  - name: tip-shear
    type: BX
    direction: [1.0]
    value: -1.0
    elements: [2]

  materials:
  - type: neohook
    name: neo-2d
    parameters:
      mu: 1.0
      lam: 1.0

  element blocks:
  - name: beam
    material: neo-2d
    elements: all
    element:
      type: EULER
      properties:
        area: 1.0
        inertia: 1.0
```
### Method of Manufactured Solution beam

```yaml
wundy:
  # Unit-length beam discretized for the manufactured solution
  nodes: [[1, 0.0], [2, 0.2], [3, 0.4], [4, 0.6], [5, 0.8], [6, 1.0]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 6]]

  boundary conditions:
    - name: left-w
      dof: W
      nodes: [1]
      value: 0.0
    - name: left-theta
      dof: THETA
      nodes: [1]
      value: 0.0
    - name: right-w
      dof: W
      nodes: [6]
      value: 0.0
    - name: right-theta
      dof: THETA
      nodes: [6]
      value: 0.0

materials:
  - type: neohook
    name: mms-neo
    parameters:
      mu: 1.0
      lam: 1.0
      E: 1.0

element blocks:
  - material: mms-neo
    name: block-1
    elements: all
    element:
      type: EULER
      properties:
        area: 1.0
        inertia: 1.0

distributed loads:
  - name: mms-load
    type: BX
    direction: [1.0]
    value: "-72.0 + 360.0 * x - 360.0 * x**2"
    elements: all
    n_gauss: 3
```