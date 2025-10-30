import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ui.schemas import DIRICHLET
from .ui.schemas import NEUMANN

logger = logging.getLogger(__name__)


def solve(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    equations: list[list[tuple[int, int, float]]],
    block_elem_map: dict[int, tuple[int, int]],
    solver: dict[str, Any],
) -> dict[str, Any]:
    solver_type = solver["type"]
    if solver_type == "DIRECT":
        return solve_direct(coords, blocks, bcs, dloads, materials, equations, block_elem_map)
    elif solver_type == "NONLINEAR":
        opts = solver.get("options") or {}
        return solve_nonlinear(
            coords, blocks, bcs, dloads, materials, equations, block_elem_map, opts
        )
    else:
        raise ValueError(f"Unknown solver {solver_type}")


def solve_direct(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    equations: list[list[tuple[int, int, float]]],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    num_dof, node_signatures, dof_map = build_dof_layout(coords, blocks, bcs)
    dofs = np.zeros(num_dof, dtype=float)
    K = np.zeros((num_dof, num_dof), dtype=float)
    F_int = np.zeros(num_dof, dtype=float)
    global_assemble(K, F_int, coords, dofs, dof_map, blocks, materials)
    F_ext = np.zeros(num_dof, dtype=float)
    assemble_global_force(F_ext, dof_map, coords, blocks, bcs, dloads, materials, block_elem_map)
    R = F_ext - F_int
    Kbc, Fbc = apply_dirichlet_bcs(K, R, dofs, bcs, dof_map)
    solution = {"stiff": K, "force": R}
    if not equations:
        dofs[:] = np.linalg.solve(Kbc, Fbc)
    else:
        Kbc, Fbc = apply_linear_constraints(Kbc, Fbc, dofs, dof_map, bcs, equations)
        x = np.linalg.solve(Kbc, Fbc)
        dofs[:num_dof] = x[:num_dof]
        solution["lagrange_mulitpliers"] = x[num_dof:]
    solution["dofs"] = dofs
    return solution


def solve_nonlinear(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    equations: list[list[tuple[int, int, float]]],
    block_elem_map: dict[int, tuple[int, int]],
    options: dict[str, Any],
) -> dict[str, Any]:
    num_dof, node_signatures, dof_map = build_dof_layout(coords, blocks, bcs)
    u = np.zeros(num_dof, dtype=float)
    K = np.zeros((num_dof, num_dof), dtype=float)
    F_int = np.zeros(num_dof, dtype=float)
    F_ext = np.zeros(num_dof, dtype=float)
    assemble_global_force(F_ext, dof_map, coords, blocks, bcs, dloads, materials, block_elem_map)

    tol: float = options.get("tolerance", 1e-8)
    maxiter: int = options.get("max iterations", 25)
    for it in range(maxiter):
        K.fill(0.0)
        F_int.fill(0.0)
        global_assemble(K, F_int, coords, u, dof_map, blocks, materials)
        rhs = F_ext - F_int
        Kbc, Fbc = apply_dirichlet_bcs(K, rhs, u, bcs, dof_map)
        if not equations:
            du = np.linalg.solve(Kbc, Fbc)
        else:
            Kbc, Fbc = apply_linear_constraints(Kbc, Fbc, u, dof_map, bcs, equations)
            x = np.linalg.solve(Kbc, Fbc)
            du = x[:num_dof]
        u += du
        if np.linalg.norm(du) < tol:
            break
    else:
        raise RuntimeError(f"Newton iterations failed to converge after {it} iterations")

    K.fill(0)
    F_int.fill(0)
    global_assemble(K, F_int, coords, u, dof_map, blocks, materials)
    solution = {"stiff": K, "force": F_ext, "dofs": u}
    return solution


def build_dof_layout(
    coords: NDArray[float], blocks: list[dict], bcs: list[dict]
) -> tuple[int, NDArray[int], NDArray[int]]:
    num_node: int = coords.shape[0]
    max_dof_per_node = len(blocks[0]["element"]["properties"]["node_freedoms"][0])
    node_signatures: NDArray[int] = np.zeros((num_node, max_dof_per_node), dtype=int)

    for block in blocks:
        node_freedoms = np.asarray(block["element"]["properties"]["node_freedoms"], dtype=int)
        for nodes in block["connect"]:
            node_signatures[nodes, :] |= node_freedoms

    # Check for dummy nodes that are not included in element connectivity
    # All dummy nodes must have associated Dirichlet BCs.  We can use the BC to fill in the node
    # signature
    for bc in bcs:
        node_signatures[bc["nodes"], bc["local_dof"]] |= 1

    if np.any(node_signatures.sum(axis=1) == 0):
        missing = np.where(node_signatures.sum(axis=1) == 0)[0]
        raise ValueError(f"Dummy node without Dirichlet BC: {missing.tolist()}")

    active_mask: NDArray[bool] = node_signatures == 1
    num_dof: int = int(active_mask.sum())

    mask = active_mask.ravel()
    map = -np.ones_like(mask, dtype=int)
    map[mask] = np.arange(num_dof)
    global_dof_map = map.reshape(node_signatures.shape)

    return num_dof, node_signatures, global_dof_map


def element_freedom_table(dof_map, nodes, node_freedoms):
    eft = [
        dof_map[node, j]
        for n, node in enumerate(nodes)
        for j, active in enumerate(node_freedoms[n])
        if active
    ]
    return eft


def global_assemble(
    K: NDArray[float],
    F_int: NDArray[float],
    coords: NDArray[float],
    u: NDArray[float],
    dof_map: NDArray[int],
    blocks: list[dict],
    materials: dict[str, Any],
) -> None:
    # Assemble global stiffness
    for block in blocks:
        properties = block["element"]["properties"]
        material = materials[block["material"]]
        for nodes in block["connect"]:
            ke, fe = get_element_state(coords[nodes], u[nodes], block["element"], material)
            eft = element_freedom_table(dof_map, nodes, properties["node_freedoms"])
            K[np.ix_(eft, eft)] += ke
            F_int[np.ix_(eft)] += fe


def assemble_global_force(
    F: NDArray[float],
    dof_map: NDArray[int],
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> None:
    apply_neumann_bcs(F, dof_map, bcs)
    apply_dloads(F, dof_map, coords, blocks, dloads, materials, block_elem_map)


def apply_dloads(
    F: NDArray[float],
    dof_map: NDArray[int],
    coords: NDArray[float],
    blocks: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> None:
    # Apply distributed loads
    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")
        for eid in dload["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dload['name']} not found in any element block"
                )
            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            properties = block["element"]["properties"]
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                A = properties["area"]
                mat = materials[block["material"]]
                rho = mat["density"]
                q = rho * A * dload["value"] * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")
            eft = element_freedom_table(dof_map, nodes, properties["node_freedoms"])
            fe = element_force(xe, q, block["element"])
            F[eft] += fe


def extract_dirichlet(
    bcs: list[dict], dof_map: NDArray[int]
) -> tuple[NDArray[int], NDArray[float]]:
    dofs: list[int] = []
    vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = dof_map[n, bc["local_dof"]]
                dofs.append(I)
                vals.append(bc["value"])
    return np.array(dofs, dtype=int), np.array(vals, dtype=float)


def apply_dirichlet_bcs(
    K: NDArray[float],
    F: NDArray[float],
    u: NDArray[float],
    bcs: list[dict],
    dof_map: NDArray[int],
) -> tuple[NDArray[float], NDArray[float]]:
    dofs, vals = extract_dirichlet(bcs, dof_map)
    Kbc = K.copy()
    Fbc = F.copy()
    ubc = np.zeros_like(vals)
    for i, dof in enumerate(dofs):
        ubc[i] = vals[i] - u[dof]
        Fbc -= K[:, dof] * ubc[i]
        Kbc[:, dof] = Kbc[dof, :] = 0.0
        Kbc[dof, dof] = 1.0
    Fbc[dofs] = ubc
    return Kbc, Fbc


def apply_neumann_bcs(F: NDArray[float], dof_map: NDArray[int], bcs: list[dict]) -> None:
    # Apply Neumann boundary conditions to force
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = dof_map[n, bc["local_dof"]]
                F[I] += bc["value"]


def apply_dirichlet_bcs_elim(
    K: NDArray[float],
    F: NDArray[float],
    dofs: NDArray[int],
    vals: NDArray[float],
) -> tuple[NDArray[float], NDArray[float], list[int]]:
    """Apply Dirchlet boundary conditions using a symmetry preserving elimination
    Let

        Ku = f

    split dofs into two sets:

        1. free
        2. prescribed

    Set up new system:

        ⎡ K_ff   K_fp ⎤ ⎧ u_f ⎫   ⎧ F_f ⎫
        ⎢             ⎥ ⎨     ⎬ = ⎨     ⎬
        ⎣ K_pf   K_pp ⎦ ⎩ u_p ⎭   ⎩ F_p ⎭

    Eliminate prescribed dofs:

        [K_ff]{u_f} = {Ff} - [K_fp]{u_p}

    """
    all_dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, dofs)]
    Ff = F[free_dofs] - np.dot(Kfp, vals)
    return Kff, Ff, free_dofs


def apply_linear_constraints(
    K: NDArray[float],
    F: NDArray[float],
    u: NDArray[float],
    dof_map: NDArray[int],
    bcs: list[dict],
    equations: list[list[tuple[int, int, float]]],
) -> tuple[NDArray[float], NDArray[float]]:
    """Enforce homogeneous linear constraints using Lagrange multiplier method, while correctly
    handling Dirichlet dofs such as dummy nodes.

    Procuedure
    ----------

    The standard augmented Lagrange system for a set of linear constraints

        C.u = r

    is written as

        ⎡ K   C.T⎤ ⎧ u ⎫   ⎧ F ⎫
        ⎢        ⎥ ⎨   ⎬ = ⎨   ⎬
        ⎣ C    0 ⎦ ⎩ 𝜆 ⎭   ⎩ r ⎭

    where:
    - K is the global stiffness
    - C is the constraint matrix
    - 𝜆 are the Lagrange multipliers enforcing the constraings
    - F is the external force vector
    - r is the rhs of the constraint.  Only homogeneous linear constraints are supported, so r is
      initially 0

    If any DOFs participating in the constraint equations are prescribed (known), they must be
    eliminated before assembling the augmented system.

    For example, if

        u_1 - u_5 = 0

    and u_1 is prescribed (∆), this becomes:

        -u_5 = -∆

    The corresponding row of C has the column for DOF 1 zeroed and r modified by -C[i, 1] * ∆

    The augmented system becomes

        ⎡ K_ff   C_f.T⎤ ⎧ u_f ⎫   ⎧ F_f ⎫
        ⎢             ⎥ ⎨     ⎬ = ⎨     ⎬
        ⎣ C_f     0   ⎦ ⎩  𝜆  ⎭   ⎩  r  ⎭

    """
    assert len(equations) > 0
    # Build the linear constrain matrix
    m = len(equations)
    n = K.shape[0]
    C: NDArray[float] = np.zeros_like(K, shape=(m, n))
    for i, equation in enumerate(equations):
        for node, dof, coeff in equation:
            I = dof_map[node, dof]
            C[i, I] = coeff

    # Gather prescribed DOFs and values
    prescribed_dofs, prescribed_vals = extract_dirichlet(bcs, dof_map)
    r: NDArray[float] = np.zeros(m, dtype=float)
    for i, row in enumerate(C):
        for j, dof in enumerate(prescribed_dofs):
            coeff = row[dof]
            if abs(coeff) > 0.0:
                r[i] -= coeff * (prescribed_vals[j] - u[dof])
                C[i, dof] = 0.0

    # Augmented system for free DOFs + Lagrange multipliers
    Ka: NDArray[float] = np.zeros_like(K, shape=(n + m, n + m))
    Fa: NDArray[float] = np.zeros_like(K, shape=(n + m,))

    # Fill blocks
    Ka[:n, :n] = K
    Ka[:n, n:] = C.T
    Ka[n:, :n] = C
    Fa[:n] = F
    Fa[n:] = r

    return Ka, Fa


def material_stiffness(material: dict[str, Any]) -> NDArray[float]:
    E = material["parameters"]["E"]
    return np.array([[E]])


def gauss_info(npoint: int) -> tuple[NDArray[float], NDArray[float]]:
    if npoint == 1:
        return (np.array([0.0]), np.array([2.0]))
    elif npoint == 2:
        xi = 1.0 / np.sqrt(3.0)
        return np.array([-xi, xi]), np.ones(2, dtype=float)
    else:
        raise NotImplementedError(f"gauss_info only supports npoint = 1 and 2, not {npoint}")


def shape(xi: float) -> NDArray[float]:
    return np.array([1.0 - xi, 1.0 + xi]) / 2.0


def shapegrad(xi: float) -> NDArray[float]:
    return np.array([-1.0, 1.0]) / 2.0


def get_element_state(
    xe: NDArray[float],
    ue: NDArray[float],
    spec: dict[str, Any],
    material: dict[str, Any],
    ngauss: int = 2,
) -> tuple[NDArray[float], NDArray[float]]:
    if spec["type"] == "T1D2":
        return get_link_state(xe, ue, spec, material, ngauss=ngauss)
    else:
        raise ValueError(f"Unknown element type {spec['type']}")


def get_link_state(
    xe: NDArray[float],
    ue: NDArray[float],
    spec: dict[str, Any],
    material: dict[str, Any],
    ngauss: int = 2,
) -> tuple[NDArray[float], NDArray[float]]:
    he = xe[1, 0] - xe[0, 0]
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected")
    A = spec["properties"]["area"]
    gp, wp = gauss_info(ngauss)
    ke = np.zeros((2, 2), dtype=float)
    fe = np.zeros(2, dtype=float)
    for i in range(ngauss):
        dNdxi = shapegrad(gp[i])
        dxdxi = np.dot(dNdxi, xe)
        dxidx = 1.0 / dxdxi.item()
        dNdx = dNdxi * dxidx
        B = dNdx[np.newaxis, :]
        D = material_stiffness(material)
        ke += wp[i] * A * np.dot(B.T, np.dot(D, B)) * dxdxi
        e = np.dot(B, ue)
        s = D[0, 0] * e
        fe += wp[i] * A * np.dot(B.T, s).ravel() * dxdxi
    return ke, fe


def element_force(
    xe: NDArray[float], q: float, spec: dict[str, Any], ngauss: int = 2
) -> NDArray[float]:
    if spec["type"] == "T1D2":
        return link_force(xe, q, spec, ngauss=ngauss)
    else:
        raise ValueError(f"Unknown element type {spec['type']}")


def link_force(
    xe: NDArray[float], q: float, spec: dict[str, Any], ngauss: int = 2
) -> NDArray[float]:
    fe = np.zeros(2, dtype=float)
    gp, wp = gauss_info(ngauss)
    for i in range(ngauss):
        xi = gp[i]
        N = shape(xi)
        dNdxi = shapegrad(xi)
        dxdxi = np.dot(dNdxi, xe)
        fe += wp[i] * np.dot(N.T, q).ravel() * dxdxi
    return fe
