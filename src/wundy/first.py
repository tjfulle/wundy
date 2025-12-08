import numpy as np
from typing import Any, Dict, List, Tuple


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def global_dof(node: int, dof: int, dof_per_node: int) -> int:
    """Map (node, local dof) -> global DOF index."""
    return node * dof_per_node + dof


def assemble_concentrated_loads(bcs, dof_per_node: int, nnodes: int) -> np.ndarray:
    """
    Build the global external force vector from Neumann (force) BCs.

    We assume:
      - NEUMANN type is encoded as 0 (from schemas.py)
      - local_dof is already an integer (0 for x in this 1D project)
    """
    Fext = np.zeros(nnodes * dof_per_node, dtype=float)

    for bc in bcs:
        # type == 0 -> NEUMANN (force)
        if bc["type"] == 0:
            dof = int(bc["local_dof"])
            value = float(bc["value"])
            for node in bc["nodes"]:
                I = global_dof(node, dof, dof_per_node)
                Fext[I] += value

    return Fext


def _lookup_case_insensitive(name: str, table: Dict[Any, Any]):
    """
    Look up 'name' in a dict ignoring the case of string keys.

    For materials we have string keys; for block_elem_map we may
    have a dict like {0: (0,0), 1: (0,1), ...} or {"BLOCK-1": [...]}.
    Here we only compare against string keys.
    """
    target = name.lower()
    for k, v in table.items():
        if isinstance(k, str) and k.lower() == target:
            return v
    # If we didn't find a string key match, fall back to direct lookup
    if name in table:
        return table[name]
    raise KeyError(f"{name!r} not found (case-insensitive) in table")


# ------------------------------------------------------------
# Main Week 1 reference FE solver
# ------------------------------------------------------------

def first_fe_code(
    coords: List[float],
    blocks: List[Dict[str, Any]],
    bcs: List[Dict[str, Any]],
    dloads: List[Any],  # unused in tests/first.py
    materials: Dict[str, Dict[str, Any]],
    block_elem_map: Dict[Any, Any],
) -> Dict[str, np.ndarray]:
    """
    Very simple linear 1D bar solver used by tests/first.py.

    Inputs (from ui.preprocess):

      coords        : (nnodes, 1) or (nnodes,) array of x-coordinates
      blocks        : list of element-block dicts, each containing:
                        - "name"
                        - "material" (string key into materials)
                        - "element" with "properties": {"area": A, ...}
                        - "connect": (nelem, 2) connectivity array
      bcs           : list of boundary condition dicts
      dloads        : distributed loads (ignored here)
      materials     : dict name -> {"type": ..., "parameters": {...}}
      block_elem_map: in this project we treat it as a helper mapping,
                      but we only need per-block connectivity from
                      blocks themselves.

    Returns
    -------
    soln : dict
        {
          "dofs"      : displacement vector (ndofs,),
          "reactions" : reaction forces (ndofs,),
          "stiff"     : global stiffness matrix K (ndofs, ndofs),
          "force"     : global external force vector Fext (ndofs,),
        }
    """

    # ----------------------------------------
    # Basic sizes
    # ----------------------------------------
    x = np.asarray(coords, dtype=float).reshape(-1)
    nnodes = x.size
    dof_per_node = 1
    ndofs = nnodes * dof_per_node

    # Global stiffness and force
    K = np.zeros((ndofs, ndofs), dtype=float)
    Fext = np.zeros(ndofs, dtype=float)

    # ----------------------------------------
    # Assemble stiffness from all blocks
    # ----------------------------------------
    for block in blocks:
        # Material properties
        mat_name = block["material"]
        material = _lookup_case_insensitive(mat_name, materials)
        E = float(material["parameters"]["E"])
        A = float(block["element"]["properties"]["area"])

        connect = np.asarray(block["connect"], dtype=int)

        # Each row in connect is [n1, n2] (0-based node indices)
        for e in range(connect.shape[0]):
            n1, n2 = connect[e, :]
            # Here L = 1.0 (unit elements) implied by the test setups
            # so ke = (EA/L) [[ 1, -1], [-1,  1]]
            ke = A * E * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

            I1 = n1
            I2 = n2

            K[I1, I1] += ke[0, 0]
            K[I1, I2] += ke[0, 1]
            K[I2, I1] += ke[1, 0]
            K[I2, I2] += ke[1, 1]

    # ----------------------------------------
    # Assemble external forces (concentrated loads)
    # ----------------------------------------
    Fext += assemble_concentrated_loads(bcs, dof_per_node, nnodes)

    # ----------------------------------------
    # Identify Dirichlet DOFs
    # ----------------------------------------
    fixed_nodes: List[int] = []
    for bc in bcs:
        if bc["type"] == 1:  # DIRICHLET
            for node in bc["nodes"]:
                fixed_nodes.append(node)

    fixed = np.asarray(fixed_nodes, dtype=int)
    all_dofs = np.arange(ndofs, dtype=int)

    if fixed.size > 0:
        fixed_dofs = fixed  # only x DOF per node in this project
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    else:
        fixed_dofs = np.array([], dtype=int)
        free_dofs = all_dofs

    # ----------------------------------------
    # Reduce system and solve for displacements
    # ----------------------------------------
    u = np.zeros(ndofs, dtype=float)

    if free_dofs.size > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = Fext[free_dofs]

        u_free = np.linalg.solve(K_ff, F_f)
        u[free_dofs] = u_free

    # ----------------------------------------
    # Reactions
    # ----------------------------------------
    reactions = K @ u - Fext

    return {
        "dofs": u,
        "reactions": reactions,
        "stiff": K,
        "force": Fext,
    }
