# USER MANUAL — WUNDY Finite Element Framework  
Comprehensive documentation for the WUNDY 1D finite-element analysis system.

---

# 1. Purpose of WUNDY
WUNDY is a modular 1D finite-element framework designed to demonstrate:

- Linear elastic material behavior  
- Nonlinear (Neo-Hookean) bar response  
- Newton–Raphson nonlinear solving  
- YAML-driven model specification  
- Full preprocess → assembly → solve workflow  
- Automated testing of all major subsystems  

This manual provides a complete explanation of how the framework is structured, how it operates, and how to run and extend it.

---

# 2. System Architecture

```
YAML input  
     ↓  
Preprocessing (ui.py)  
     ↓  
Assembly (elements.py, materials.py)  
     ↓  
Linear or Nonlinear Solve (first.py or solver.py)  
     ↓  
Output (DOFs, reactions, stiffness, forces)
```

Each component is isolated and modular so the system can easily be extended.

---

# 3. YAML Input Format

A model is defined entirely in a YAML file containing:

- **nodes**  
- **elements**  
- **materials**  
- **element blocks**  
- **boundary conditions**  
- **concentrated loads**  
- **distributed loads**

Example:

```yaml
wundy:
  nodes: [[1,0], [2,1], [3,2]]
  elements: [[1,1,2], [2,2,3]]
  materials:
    - type: elastic
      name: MAT-1
      parameters: {E: 10.0, nu: 0.3}
  element blocks:
    - name: BLOCK-1
      material: MAT-1
      elements: all
      element: {type: T1D1, properties: {area: 1.0}}
```

All input is validated using schema rules.

---

# 4. Preprocessing (ui.py)

Preprocessing converts the YAML into arrays and mappings ready for the solver.

### What it produces:

- `coords` → matrix of nodal coordinates  
- `blocks` → list of element blocks  
- `materials` → dictionary of material definitions  
- `bcs` → boundary conditions  
- `dload` → distributed loads  
- `block_elem_map` → mapping from block to global element indices  
- `node_map` and `elem_map` → 1-based → 0-based conversions  

### Zero-Based Mapping Diagram
```
User YAML nodes: 1,2,3,4 → internal indices: 0,1,2,3
User elements: 1,2,3,4 → internal indices: 0,1,2,3
```

This ensures vector assembly works correctly.

---

# 5. Elements (elements.py)

## 5.1 T1D1 Bar Element

### Strain
```
ε = (u2 – u1) / L
```

### Stiffness Matrix (Linear)
```
k = (E*A / L) * [[ 1, -1],
                 [-1, 1]]
```

### Internal Force
```
R_int = Bᵀ σ A
B = [-1/L, 1/L]
```

The element library supports both linear and nonlinear materials.

---

# 6. Material Models (materials.py)

## 6.1 Linear Elastic
```
σ = E ε
tangent = E
```

## 6.2 Neo-Hookean (1D Demonstration)
A simple nonlinear stress law used in the nonlinear example:

```
σ = E ε (1 + α ε)
tangent = E (1 + 2 α ε)
```

This enables nonlinear solving without requiring 3D hyperelasticity.

---

# 7. Assembly Procedure

Global stiffness and force assembly follow:

```
For each element:
    Compute ke
    Compute internal forces
    Map local DOFs → global DOFs
    Insert ke into global K
```

External forces come from:

- Dirichlet BC reaction extraction  
- Concentrated loads  
- Distributed loads  

Global DOF numbering (1 DOF per node):

```
global_dof = node_index * dof_per_node + local_dof
```

---

# 8. Solvers

## 8.1 Basic Linear Solver (first.py)

Used for tests and simple models. It:

- Assembles global K  
- Applies boundary conditions  
- Applies concentrated loads  
- Solves for displacements  
- Returns DOFs, stiffness, forces, reactions  

This file validates the entire preprocessing and assembly pipeline.

---

## 8.2 Newton–Raphson Nonlinear Solver (solver.py)

Used when any material is `"type: NEOHOOKEAN"`.

Algorithm:

```
u = 0
repeat until converged:
    Assemble tangent stiffness K
    Assemble internal force R_int
    residual r = Fext - R_int
    solve K Δu = r
    u ← u + Δu
```

Linear problems converge in 1 iteration; nonlinear problems converge quadratically.

---

# 9. Examples

## Run the linear example:
```
python bin/run.py src/wundy/linear_bar.yaml
```

## Run the nonlinear example:
```
python bin/run.py src/wundy/nonlinear_bar.yaml
```

Both will print DOFs and reactions.

---

# 10. Method of Manufactured Solutions (MMS)

An MMS test verifies the full solver pipeline.

### Choose:
```
u(x) = sin(π x)
Domain: [0,1]
```

### Derive load:
```
ε = du/dx = π cos(πx)
σ = E ε
b(x) = -dσ/dx = π² E sin(πx)
```

### FE model must reproduce:
```
u_exact(x) = sin(π x)
```

The MMS test (optional extension) uses distributed loads to reproduce the analytical field.  
This validates:

- Preprocess  
- Assembly  
- Distributed load integration  
- Stiffness and internal force  
- Solver correctness  

This test is **not required by existing coursework tests** and is implemented separately so it does not affect the required test suite.

---

# 11. Automated Testing

### `test_first.py`
Validates:
- Linear solver correctness
- Stiffness matrix matching analytical form
- DOF and reaction outputs
- Case-insensitive material/block lookup

### `test_elements.py`
Validates:
- Element stiffness
- Internal forces
- Strain calculations

### `test_materials.py`
Validates:
- Linear elastic σ and tangent
- Error handling for invalid materials

### `test_user_input.py`
Validates:
- Schema correctness  
- Preprocessing correctness  
- Name normalization  
- Node/element/block consistency  

All tests must pass:

```
pytest
```

Expected:
```
8 passed
```

---

# 12. Extending WUNDY

Future improvements can include:

- Multiple DOFs per node  
- Frame/truss elements  
- Plasticity  
- Eigenvalue analysis  
- Dynamic response  
- Visualization tools  

---

# 13. Author

Developed by **Verl Grant** as a complete modular 1D finite-element solver.

