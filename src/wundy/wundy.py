from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ui.schemas import DIRICHLET
from .ui.schemas import NEUMANN

dof_per_node: int = 1


def solve(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    equations: list[list[tuple[int, int, float]]],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node
    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)
    global_assemble(K, F, coords, blocks, bcs, materials)
    apply_dloads(F, coords, blocks, dloads, materials, block_elem_map)
    dofs = np.zeros(num_dof, dtype=float)
    Kbc, Fbc = apply_bcs(K, F, bcs, dofs)
    if equations:
        Ka, Fa = apply_linear_constraints(Kbc, Fbc, equations)
        x = np.linalg.solve(Ka, Fa)
        dofs[:] = x[: K.shape[0]]
        lam = x[K.shape[0] :]
    else:
        # solve the system
        dofs[:] = np.linalg.solve(Kbc, Fbc)
    solution = {"dofs": dofs, "stiff": K, "force": F}
    return solution


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree of freedom index for a given node and local dof

    NOTE: Assumes elements have uniform degrees of freedom across the mesh.

    """
    return node * dof_per_node + local_dof


def global_assemble(
    K: NDArray[float],
    F: NDArray[float],
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    materials: dict[str, Any],
) -> None:
    # Assemble global stiffness
    for block in blocks:
        properties = block["element"]["properties"]
        material = materials[block["material"]]
        for nodes in block["connect"]:
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes]
            ke = element_stiffness(xe, material, properties)
            K[np.ix_(eft, eft)] += ke

    # Apply Neumann boundary conditions to force
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]


def apply_dloads(
    F: NDArray[float],
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
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                A = block["element"]["properties"]["area"]
                mat = materials[block["material"]]
                rho = mat["density"]
                q = rho * A * dload["value"] * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            fe = element_force(xe, q)
            F[eft] += fe


def apply_bcs(
    K: NDArray[float],
    F: NDArray[float],
    bcs: list[dict],
    u: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(bc["value"])
    Kbc, Fbc = apply_dirichlet_bcs(K, F, prescribed_dofs, prescribed_vals)
    return Kbc, Fbc


def apply_dirichlet_bcs(
    K: NDArray[float],
    F: NDArray[float],
    dofs: NDArray[int],
    vals: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
    Kbc = K.copy()
    Fbc = F.copy()
    for dof, val in zip(dofs, vals):
        Fbc -= K[:, dof] * val
        Kbc[:, dof] = 0.0
        Kbc[dof, :] = 0.0
        Kbc[dof, dof] = 1.0
        Fbc[dof] = val
    return Kbc, Fbc


def apply_dirichlet_bcs_elim(
    K: NDArray[float],
    F: NDArray[float],
    dofs: NDArray[int],
    vals: NDArray[float],
) -> tuple[NDArray[float], NDArray[float], list[int]]:
    # Apply Dirchlet boundary conditions using a symmetry preserving elimination
    # Let
    #   Ku = f
    # split dofs into two sets:
    #   1. free
    #   2. prescribed
    # Set up new system:
    #
    #  | K_ff  K_fp |  [ u_f ]   | F_f |
    #  | K_pf  K_pp |  [ u_p ]   | F_p |
    #
    # Eliminate prescribed dofs:
    #   K_ff.u_f = Ff - K_fp.u_p
    all_dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(all_dofs, dofs)
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, dofs)]
    Ff = F[free_dofs] - np.dot(Kfp, vals)
    return Kff, Ff, free_dofs


def apply_linear_constraints(
    K: NDArray[float], F: NDArray[float], equations: list[list[tuple[int, int, float]]]
) -> tuple[NDArray[float], NDArray[float]]:
    m = len(equations)
    if not m:
        return K, F
    n = K.shape[0]
    C: NDArray[float] = np.zeros_like(K, shape=(m, n))
    for i, equation in enumerate(equations):
        for node, dof, coeff in equation:
            I = global_dof(node, dof, dof_per_node)
            C[i, I] = coeff
    Ka: NDArray[float] = np.zeros_like(K, shape=(n + m, n + m))
    Fa: NDArray[float] = np.zeros_like(K, shape=(n + m,))

    # Fill blocks
    Ka[:n, :n] = K
    Ka[:n, n:] = C.T
    Ka[n:, :n] = C
    Fa[:n] = F
    Fa[n:] = 0.0  # All constraints are homogenous
    return Ka, Fa


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


def material_stiffness(material: dict[str, Any]) -> NDArray[float]:
    E = material["parameters"]["E"]
    return np.array([[E]])


def element_stiffness(
    xe: NDArray[float], material: dict[str, Any], properties: dict[str, Any], ngauss: int = 2
) -> NDArray[float]:
    he = xe[1, 0] - xe[0, 0]
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected")
    A = properties["area"]
    gp, wp = gauss_info(ngauss)
    ke = np.zeros((2, 2), dtype=float)
    for i in range(ngauss):
        dNdxi = shapegrad(gp[i])
        dxdxi = np.dot(dNdxi, xe)
        dxidx = 1.0 / dxdxi.item()
        dNdx = dNdxi * dxidx
        B = dNdx[np.newaxis, :]
        D = material_stiffness(material)
        ke += wp[i] * A * np.dot(B.T, np.dot(D, B)) * dxdxi
    return ke


def element_force(xe: NDArray[float], q: float, ngauss: int = 2) -> NDArray[float]:
    fe = np.zeros(2, dtype=float)
    gp, wp = gauss_info(ngauss)
    for i in range(ngauss):
        xi = gp[i]
        N = shape(xi)
        dNdxi = shapegrad(xi)
        dxdxi = np.dot(dNdxi, xe)
        fe += wp[i] * np.dot(N.T, q).ravel() * dxdxi
    return fe
