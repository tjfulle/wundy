from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from .model import Model


class Solver:
    type: str = "<none>"
    method: str | None = None
    subclasses: list["Solver"] = []

    def __init__(self, **options: Any) -> None: ...

    def __init_subclass__(cls):
        Solver.subclasses.append(cls)

    @classmethod
    def factory(cls, type: str, method: str | None = None, **options: Any) -> "Solver":
        for subclass in cls.subclasses:
            if subclass.type == type and subclass.method == method:
                return subclass(**options)
        raise TypeError(f"Unknown solver type {type!r} and method {method!r}")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DirectSolver(Solver):
    type: str = "DIRECT"

    def __call__(self, model: "Model") -> dict[str, Any]:
        u = np.zeros(model.num_dof, dtype=float)
        du = np.zeros(model.num_dof, dtype=float)
        K, F_int = model.assemble_system(u, du)
        F_ext = model.global_force(u, du)
        R = F_ext - F_int
        Kbc, Fbc = model.apply_dirichlet_bcs(K, R, u, du)
        solution = {"stiff": K, "force": R}
        if not model.equations:
            u[:] = np.linalg.solve(Kbc, Fbc)
        else:
            Kbc, Fbc = model.apply_linear_constraints(Kbc, Fbc, u, du)
            x = np.linalg.solve(Kbc, Fbc)
            u[: model.num_dof] = x[: model.num_dof]
            solution["lagrange_mulitpliers"] = x[model.num_dof :]
        solution["dofs"] = u
        return solution
