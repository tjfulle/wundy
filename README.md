WUNDY — 1D Finite Element Framework

A modular educational FEA code for linear and nonlinear bar elements, developed step-by-step for the course final project.

Overview

WUNDY is a lightweight, modular 1D finite-element framework designed to demonstrate:

Linear elastic bar elements

Nonlinear Neo-Hookean bar response

Newton–Raphson nonlinear solving

YAML-driven input files

Full preprocess → assembly → solve pipeline

Automated verification using pytest

Method of Manufactured Solutions (MMS) accuracy test

All models are defined in YAML, validated by the input schema, converted to FE arrays, assembled into global stiffness and force vectors, and solved by the linear or nonlinear solver.

Repository Structure

wundy/
├── bin/
│ └── run.py
│
├── src/wundy/
│ ├── first.py
│ ├── elements.py
│ ├── materials.py
│ ├── ui.py
│ ├── solver.py
│ ├── linear_bar.yaml
│ ├── nonlinear_bar.yaml
│ └── examples/
│ ├── mms_input.yaml
│ └── run_mms.py
│
├── tests/
│ ├── first.py
│ ├── test_elements.py
│ ├── test_materials.py
│ ├── test_user_input.py
│ └── test_mms.py
│
└── README.md

Features Implemented
1. YAML Input System

Each FE model is defined by a single YAML file including:

nodes

elements

node sets / element sets

boundary conditions

materials

element blocks

concentrated loads

distributed loads

The ui.preprocess() function converts YAML into solver-ready arrays.

2. Element Library (T1D1)

Implemented element:

T1D1 — 2-node, 1D bar element

Capabilities:

strain computation

linear stiffness matrix

nonlinear tangent stiffness

internal force vector

equivalent nodal force load generation

3. Material Models

Linear Elastic:
σ = E ε

Neo-Hookean (1D demonstration):
Used to test nonlinear Newton iterations.

4. Assembly System

The solver assembles:

global stiffness

global internal force

global external load vector

DOF numbering rule:
global_dof = node_index * dof_per_node + local_dof

(Only one DOF per node in this project.)

5. Nonlinear Newton Solver

Algorithm:

u = 0
repeat:
• assemble tangent stiffness
• compute internal force
• compute residual
• solve for Δu
• update u
until converged

Automatically used whenever a nonlinear material is defined.

Running Examples
Linear Bar

python bin/run.py src/wundy/linear_bar.yaml

Nonlinear Bar

python bin/run.py src/wundy/nonlinear_bar.yaml

Method of Manufactured Solutions (MMS)

WUNDY includes a full MMS verification using the exact solution:

u_exact(x) = sin(pi x)

The forcing is pre-generated in:

src/wundy/examples/mms_input.yaml

To run MMS:

python -m wundy.examples.run_mms

The output includes:

FE displacement

exact displacement

nodal error

max error

L2 error

This verifies correctness of assembly and solution.

Tests

Run all tests:

pytest

Expected result:

9 passed

Tests cover:

materials

elements

preprocess

user input

linear solver

distributed loads

MMS accuracy

Development Timeline

Week 1: Basic linear solver (first.py)
Week 2: Elements, materials, preprocessing
Week 3: Neo-Hookean material + Newton solver
Final: MMS, nonlinear bar, full test suite

Requirements

numpy
pyyaml
schema
pytest

Install packages:

pip install -r requirements.txt

Notes

Developed by Verl Grant as a complete instructional 1D finite-element codebase.
All components are fully functional and validated through automated tests and MMS verification.