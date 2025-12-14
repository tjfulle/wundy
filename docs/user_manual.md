# Wundy User Manual 

## 1. Introduction

**Wundy** is a one-dimensional finite element (FE) solver for:

- Linear axial bar problems,
- A nonlinear 1D Neo-Hooke bar solved with Newton’s method,
- Euler–Bernoulli beam bending in a single plane.

The code assembles the global stiffness matrix, applies boundary
conditions, processes concentrated and distributed loads, and returns
the global displacement vector, stiffness matrix, and force vector.
**Wundy** is a one-dimensional finite element (FE) solver for:

- Linear axial bar problems,
- A nonlinear 1D Neo-Hooke bar solved with Newton’s method,
- Euler–Bernoulli beam bending in a single plane.

The code assembles the global stiffness matrix, applies boundary
conditions, processes concentrated and distributed loads, and returns
the global displacement vector, stiffness matrix, and force vector.

This manual describes:

- The physical problems solved by Wundy,
- The structure of a valid YAML input file,
- Node, element, material, boundary-condition, and load syntax,
- The preprocessing rules applied before assembly,
- The available global solvers.
- The physical problems solved by Wundy,
- The structure of a valid YAML input file,
- Node, element, material, boundary-condition, and load syntax,
- The preprocessing rules applied before assembly,
- The available global solvers.

---

## 2. Physical Problem Description

### 2.1 Linear Axial Bar

Wundy solves linear, static, one-dimensional bar problems of the form:

d/dx ( E * A * du/dx ) + q(x) = 0

where:

- E — Young's modulus  
- A — cross-sectional area  
- u(x) — axial displacement  
- q(x) — distributed axial load per unit length

The domain is discretized into 2-node T1D1 bar elements with linear shape functions.

### 2.2 Nonlinear Neo-Hooke Bar

For the Neo-Hooke bar, the solver `newton_bar_neo_hooke_1d` uses a 1D
first Piola–Kirchhoff stress law of the form:

P(F) = (E/2) * (F - 1/F)

where F is the axial stretch.

At small strain (when F is close to 1), the tangent dP/dF approaches E, so the Neo-Hooke response
matches linear elasticity in the small-strain limit.

A Newton–Raphson iteration is used to solve the nonlinear equilibrium equations.

### 2.3 Euler–Bernoulli Beam

For bending, Wundy uses a 2-node Euler–Bernoulli beam element with
transverse displacement w(x) and rotation theta(x) at each node.

The element has 4 degrees of freedom:

[w1, theta1, w2, theta2]^T

The standard 4x4 Euler–Bernoulli stiffness matrix is:

k_e = (E * I / L^3) *
[
  [ 12,      6L,    -12,      6L   ],
  [  6L,   4L^2,     -6L,    2L^2   ],
  [ -12,     -6L,     12,     -6L   ],
  [  6L,   2L^2,     -6L,    4L^2   ]
]

where:

- E is Young's modulus  
- I is the second moment of area  
- L is the element length

For a uniform transverse load q, the consistent nodal load vector is:

f_e =
[
  q*L/2,
  q*L^2/12,
  q*L/2,
 -q*L^2/12
]

---

### 2.4 Assumptions
- \( E \) — Young’s modulus  
- \( A \) — cross-sectional area  
- \( u(x) \) — axial displacement  
- \( q(x) \) — distributed axial load per unit length.

The domain is discretized into 2-node `T1D1` bar elements with
linear shape functions.

### 2.2 Nonlinear Neo-Hooke Bar

For the Neo-Hooke bar, the solver `newton_bar_neo_hooke_1d` uses a 1D
first Piola–Kirchhoff stress law of the form

\[
P(F) = \frac{E}{2} \left(F - \frac{1}{F}\right),
\]

with \( F \) the axial stretch. At small strain (near \(F=1\)), the
tangent \( dP/dF \) reduces to \( E \), so the Neo-Hooke response
matches linear elasticity in the limit. A Newton–Raphson iteration is
used to solve the nonlinear equilibrium equations.

### 2.3 Euler–Bernoulli Beam

For bending, Wundy uses a 2-node Euler–Bernoulli beam element with
transverse displacement \( w(x) \) and rotation \( \theta(x) \) at each
node. The element has 4 DOFs:

\[
[w_1,\, \theta_1,\, w_2,\, \theta_2]^T.
\]

The standard 4×4 Euler–Bernoulli stiffness matrix is used:

\[
k_e = \frac{EI}{L^3}
\begin{bmatrix}
12   & 6L   & -12 & 6L \\
6L   & 4L^2 & -6L & 2L^2 \\
-12  & -6L  & 12  & -6L \\
6L   & 2L^2 & -6L & 4L^2
\end{bmatrix},
\]

where \( E \) is Young’s modulus, \( I \) is the second moment of area,
and \( L \) is the element length.

For a uniform transverse load \( q \), the consistent nodal load vector
is

\[
f_e =
\begin{bmatrix}
qL/2 \\
qL^2/12 \\
qL/2 \\
-qL^2/12
\end{bmatrix}.
\]

### Assumptions

The solver assumes:

1. **1D domain** with nodes on the real line.
2. **Two-node bar or beam elements** (called `T1D1`).
3. **One or two degrees of freedom per node for bar and beam elements respectively** (axial displacement in the x-direction).
4. **Linear elastic and Neo-Hookean materials**.
5. **Small-strain linear kinematics**.
6. **YAML input files** fully validated against the schema before assembly.

---

## 3. Input Files

### 3.1 Overview of the Input File
## 3. Input Files

### 3.1 Overview of the Input File

A valid input file is a YAML document with one root key, here is a template for such an input file:

```yaml
wundy:
  nodes:
    - [node_id, x_coord]

  elements:
    - [element_id, node1_id, node2_id]

  materials:
    - type: material_type
      name: material_name
      parameters:
        E: Youngs_modulus
        nu: Poissons_ratio
      density: material_density

  element blocks:
    - name: block_name
      material: material_name
      elements: element_set_name_or_ids
      element:
        type: T1D1
        properties:
          area: cross_section_area
          I : moment_of_inertia
          I : moment_of_inertia

  boundary conditions:
    - nodes: node_ids_or_node_set
      dof: x_or_y
      dof: x_or_y
      value: displacement_or_force
      type: DIRICHLET_or_NEUMANN

  node sets:
    - name: node_set_name
      nodes: [node_id, node_id]

  element sets:
    - name: element_set_name
      elements: [element_id, element_id]

  concentrated loads:
    - nodes: node_ids_or_node_set
      dof: x_or_y_
      dof: x_or_y_
      value: nodal_force_value

  distributed loads:
    -name: dload_1
    - type: load_type   # BX, GRAV, or QY
    -name: dload_1
    - type: load_type   # BX, GRAV, or QY
      elements: element_set_or_ids
      value: load_magnitude #Needed for UNIFORM profile and QY
      profile : UNIFORM_or_EQUATION #Optional for BX and GRAV 
      expression : "a*x**2 + b*x" #Only for profile: EQUATION,Python expression in the variable x, evaluated at the Gauss points 
      value: load_magnitude #Needed for UNIFORM profile and QY
      profile : UNIFORM_or_EQUATION #Optional for BX and GRAV 
      expression : "a*x**2 + b*x" #Only for profile: EQUATION,Python expression in the variable x, evaluated at the Gauss points 
      direction: [±1.0]
```

### 3.2 Processing of the Input File

Internally, the preprocessor creates entries like:

{
    "name": ...,
    "elements": [local element indices],
    "type": "BX" / "GRAV" / "QY",
    "direction": [float],
    "value": float,          # if given
    "profile": "UNIFORM" or "EQUATION",
    "expression": str,       # if EQUATION
}

The apply_distributed_loads routine then:

  For BX + UNIFORM:
    uses the standard 2-node bar equivalent nodal load formula.
  For BX + EQUATION:
    calls element_external_force_t1d1_arbitrary with a Python q_func(x).
  For GRAV:
    treats it similarly to a uniform BX load.
  For QY:
    uses element_load_euler_bernoulli_uniform on beam elements.

---

## 4. Global Solvers

### 4.1 Linear Axial Bar: first_fe_code

Signature: first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)

Assembles the global stiffness matrix for axial DOFs (1 per node).

Applies Dirichlet and Neumann boundary conditions.

Adds equivalent nodal forces from BX/GRAV distributed loads.

Returns a dictionary with at least:

"dofs" – global displacement vector,

"stiff" – global stiffness matrix,

"force" – global force vector.

### 4.2 Linear Axial Bar: Nonlinear Neo-Hooke Bar: newton_bar_neo_hooke_1d

Signature: newton_bar_neo_hooke_1d(coords, blocks, bcs, dloads, materials, block_elem_map)

Uses the Neo-Hooke 1D material law,

Iterates with Newton–Raphson,

Returns a solution structure similar to first_fe_code.

For small loads, tests verify that the Neo-Hooke solution is close to
the linear-elastic bar solution.

### 4.3 Euler–Bernoulli Beam: beam_fe_code

beam_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)

Signature: beam_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)

Interprets each T1D1 element with properties area and I as an
Euler–Bernoulli beam element.

Each node carries two DOFs:

DOF 0 (dof: x in YAML): transverse displacement w

DOF 1 (dof: y in YAML): rotation theta

Assembles the global beam stiffness matrix using
element_stiffness_euler_bernoulli.

Adds consistent nodal loads from QY distributed loads via
element_load_euler_bernoulli_uniform.

Applies Dirichlet/Neumann boundary conditions on 
𝑤 and θ.

Returns:

dofs – global vector [w1, theta1, w2, theta2, ...],

stiff – global beam stiffness matrix,

force – global right-hand side vector after load assembly.

---

## 5.Testing and Manufactured Solutions

The test suite includes:

Element-level tests:

Bar stiffness via Gauss integration vs. analytic (EA/L) matrix,

Internal forces for a simple bar displacement state,

Gauss quadrature points and weights,

Euler–Bernoulli stiffness matrix vs. the textbook 4×4 formula,

Euler–Bernoulli consistent nodal load for uniform q.

Material tests:

Neo-Hooke tangent at 𝐹=1 equals E,

Neo-Hooke differs from linear elasticity at finite strain.

Global bar tests:

Axial bar with a point load,

Axial bar with uniform distributed load,

Neo-Hooke bar vs. linear bar for small loads.

Manufactured solution tests:

Element-level arbitrary line load q(x)=x**2 integrated via Gauss rule,

Global beam; cantilever subjected to uniform QY load with
analytic reference deflections.

These tests serve as method-of-manufactured-solution checks and validate
both the local element implementations and the global assembly.

---