import numpy as np
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------
@dataclass
class SolverOptions:
    max_iter: int = 25
    tol: float = 1e-10
    verbose: bool = False


# ----------------------------------------------------------------------
# Assembly helpers
# ----------------------------------------------------------------------
def assemble_global_stiffness(pre):
    coords = pre["coords"]
    blocks = pre["blocks"]
    mats = pre["materials"]

    ndof = coords.shape[0]
    K = np.zeros((ndof, ndof))

    for blk in blocks:
        conn = blk["connect"]
        matname = blk["material"]
        mat = mats[matname]
        E = mat["parameters"]["E"]
        A = blk["element"]["properties"]["area"]

        for (n1, n2) in conn:
            x1 = coords[n1, 0]
            x2 = coords[n2, 0]
            L = x2 - x1
            k = (A * E) / L

            ke = np.array([[ k, -k],
                           [-k,  k]])

            dofs = [n1, n2]
            for i in range(2):
                for j in range(2):
                    K[dofs[i], dofs[j]] += ke[i, j]

    return K


def assemble_internal_force(pre, u):
    K = assemble_global_stiffness(pre)
    return K @ u


def assemble_external_force(pre):
    coords = pre["coords"]
    ndof = coords.shape[0]
    F = np.zeros(ndof)

    # dx for MMS load integration
    if ndof > 1:
        dx = coords[1, 0] - coords[0, 0]
    else:
        dx = 1.0

    # ------------------------------------------------------------
    # NEUMANN loads: convert value = -f  → nodal force = +f dx
    # ------------------------------------------------------------
    for bc in pre["bcs"]:
        if bc["type"] == 0:  # NEUMANN
            for n in bc["nodes"]:
                F[n] += -bc["value"] * dx   # <----- THE FIX

    # Distributed loads
    for dl in pre.get("dload", []):
        for e in dl["elements"]:
            blk = None
            for b in pre["blocks"]:
                if e in b["elem_map"]:
                    blk = b
                    break

            le = blk["elem_map"][e]
            n1, n2 = blk["connect"][le]
            x1 = coords[n1, 0]
            x2 = coords[n2, 0]
            L = x2 - x1

            q = dl["value"]
            fe = q * L / 2 * np.array([1, 1])

            F[n1] += fe[0]
            F[n2] += fe[1]

    return F


def apply_dirichlet(K, R, pre):
    # DIRICHLET = 1
    for bc in pre["bcs"]:
        if bc["type"] == 1:
            for n in bc["nodes"]:
                val = bc["value"]

                K[n, :] = 0.0
                K[:, n] = 0.0
                K[n, n] = 1.0
                R[n] = val

    return K, R


# ----------------------------------------------------------------------
# Newton solver
# ----------------------------------------------------------------------
def newton_solve_1d(pre, options: SolverOptions = None):
    if options is None:
        options = SolverOptions()

    ndof = pre["coords"].shape[0]
    u = np.zeros(ndof)

    F_ext = assemble_external_force(pre)

    for it in range(options.max_iter):
        K = assemble_global_stiffness(pre)
        R_int = assemble_internal_force(pre, u)

        residual = F_ext - R_int

        K2, r2 = apply_dirichlet(K.copy(), residual.copy(), pre)

        du = np.linalg.solve(K2, r2)
        u += du

        if np.linalg.norm(du) < options.tol:
            return {
                "converged": True,
                "dofs": u,
                "stiff": K,
                "residual": residual,
            }

    return {
        "converged": False,
        "dofs": u,
        "stiff": K,
        "residual": residual,
    }
