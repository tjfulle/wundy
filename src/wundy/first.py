from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN

# ---------------------------------------------------------------------------
# Material response (linear elastic bar)
# ---------------------------------------------------------------------------


def material_tangent_stiffness_elastic(material: dict[str, Any]) -> float:
    """Return the 1D elastic tangent stiffness (Young's modulus).

    Parameters
    ----------
    material:
        Material dictionary with a "parameters" sub-dictionary containing
        the key "E" for Young's modulus.

    Returns
    -------
    float
        Young's modulus E.
    """
    return float(material["parameters"]["E"])


def material_stress_elastic(material: dict[str, Any], strain: float) -> float:
    """Compute Cauchy stress for a 1D linear elastic material.

    Parameters
    ----------
    material:
        Material dictionary compatible with
        material_tangent_stiffness_elastic.
    strain:
        Axial strain (epsilon).

    Returns
    -------
    float
        Axial stress sigma = E * strain.
    """
    E = material_tangent_stiffness_elastic(material)
    return E * strain


def lame_parameters_from_E_nu(material: dict[str, Any]) -> tuple[float, float]:
    """Compute Lame parameters (lambda, mu) from E and nu for an isotropic material.

    The material dictionary is expected to contain a "parameters" mapping
    with entries "E" (Young's modulus) and "nu" (Poisson's ratio).
    These are converted to the Lame constants using

        mu     = E / (2 * (1 + nu))
        lambda = E * nu / ((1 + nu) * (1 - 2 * nu))

    Parameters
    ----------
    material:
        Material dictionary with "parameters" containing "E" and "nu".

    Returns
    -------
    (lambda_, mu)
        Tuple of Lame parameters as floats.
    """
    params = material["parameters"]
    E = float(params["E"])
    nu = float(params["nu"])
    mu = E / (2.0 * (1.0 + nu))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lambda_, mu


def material_neo_hooke_1d_pk1(
    material: dict[str, Any],
    F: float,
) -> tuple[float, float]:
    """Return 1D Neo-Hookean stress and tangent for the class model.

    In this project we use the 1D Neo-Hooke law

        sigma(F) = (E / 2) * (F - 1 / F),

    where F is the stretch (1 + strain). In 1D we identify the scalar
    first Piola–Kirchhoff stress with sigma.

    Differentiating with respect to F gives the algorithmic tangent

        d sigma / dF = (E / 2) * (1 + 1 / F^2).

    A Taylor expansion around F = 1 shows that this model linearizes to
    standard 1D linear elasticity with modulus E:

        sigma(F) ≈ E * (F - 1)  for F close to 1.

    Parameters
    ----------
    material:
        Material dictionary with "parameters" containing "E" (Young's
        modulus). Any "nu" entry is ignored by this 1D model.
    F:
        Stretch (deformation gradient in 1D). Must be positive.

    Returns
    -------
    (P, dP_dF)
        Tuple containing the 1D stress P and its derivative with respect
        to F according to the formulas above.
    """
    if F <= 0.0:
        raise ValueError(
            f"Neo-Hooke 1D requires F > 0, got F = {F}. "
            "Check your boundary conditions and initial guess."
        )

    E = float(material["parameters"]["E"])
    inv_F = 1.0 / F
    inv_F2 = inv_F * inv_F

    # sigma(F) = (E/2) (F - F^{-1})
    P = 0.5 * E * (F - inv_F)

    # d sigma / dF = (E/2) (1 + F^{-2})
    dP_dF = 0.5 * E * (1.0 + inv_F2)

    return float(P), float(dP_dF)


# ---------------------------------------------------------------------------
# Element response for T1D1 (two-node 1D bar) with Gauss quadrature
# ---------------------------------------------------------------------------


def gauss_points_1d_two_point() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return abscissae and weights for 2-point Gauss quadrature on [-1, 1].

    This rule is exact for polynomials up to degree three. For the 2-node
    bar element used here the strain-displacement matrix is constant, so the
    integrands for stiffness and internal force are at most quadratic and
    are therefore integrated exactly by this rule.
    """
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    w = np.ones(2, dtype=float)
    return xi, w


def shape_functions_t1d1(xi: float) -> NDArray[np.float64]:
    """Return shape functions N1, N2 for the 2-node 1D bar at xi.

    The reference element is the interval xi in [-1, 1] with the usual
    linear shape functions

        N1(xi) = (1 - xi) / 2
        N2(xi) = (1 + xi) / 2

    Parameters
    ----------
    xi:
        Gauss point in the reference coordinate system.

    Returns
    -------
    ndarray
        Array [N1, N2] evaluated at xi.
    """
    return np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)


def shape_function_derivatives_t1d1() -> NDArray[np.float64]:
    """Return dN/dxi for the 2-node 1D bar (constant over the element).

    On the reference element xi in [-1, 1] the derivatives of the shape
    functions with respect to the reference coordinate xi are

        dN1/dxi = -1/2
        dN2/dxi = +1/2

    Returns
    -------
    ndarray
        Array [dN1_dxi, dN2_dxi].
    """
    return np.array([-0.5, 0.5], dtype=float)


def element_stiffness_t1d1(
    xe: NDArray[np.float64],
    area: float,
    E: float,
) -> NDArray[np.float64]:
    """Compute the 2x2 stiffness matrix for a T1D1 bar element with Gauss quadrature.

    The element is treated as a 2-node isoparametric bar on the reference
    interval xi in [-1, 1]. The stiffness matrix is obtained from

        k_e = integral of [B(xi)^T * E * A * B(xi) * J] over xi in [-1, 1],

    where B is the strain–displacement vector and J is the Jacobian
    (here J = he / 2, the half element length). For the linear T1D1
    element B is constant, and the 2-point Gauss rule integrates this
    expression exactly.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape (2, num_dim).
        Only the first column xe[:, 0] (the x-coordinate) is used.
    area:
        Cross-sectional area A of the bar.
    E:
        Young’s modulus E.

    Returns
    -------
    ndarray
        Element stiffness matrix k_e with shape (2, 2).
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_stiffness_t1d1")

    dN_dxi = shape_function_derivatives_t1d1()
    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0  # dx/dxi for the linear mapping

    ke = np.zeros((2, 2), dtype=float)
    for a in range(xi_gp.size):
        B = dN_dxi / J  # dN/dx (constant here, but written in Gauss form)
        ke += (B[:, None] @ B[None, :]) * (E * area * J * w_gp[a])

    return ke


def element_kinematics_t1d1(
    xe: NDArray[np.float64],
    ue: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Compute basic 1D kinematic quantities at Gauss points for a T1D1 element.

    This helper is the bridge from nodal displacements to finite-strain
    quantities. It uses the 2-point Gauss rule on the reference interval
    xi in [-1, 1] to compute, at each Gauss point,

    * the deformation gradient F,
    * the Green–Lagrange strain E = (F^2 - 1) / 2, and
    * the strain–displacement vector B = du/dX.

    For the linear 2-node bar these fields are constant along the element,
    but we still return arrays with one value per Gauss point so that the
    same interface can be used for more complex elements or materials.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape (2, num_dim).
        Only xe[:, 0] is used as the reference coordinate X.
    ue:
        Element displacement vector [u1, u2] with shape (2,).

    Returns
    -------
    F_gp:
        Array of deformation gradients at the Gauss points (length 2).
    E_gp:
        Array of Green–Lagrange strains at the Gauss points (length 2).
    B:
        Strain–displacement vector B with shape (2,) (same for all Gauss
        points in this element).
    J:
        Jacobian J = dX/dxi = he / 2 for the mapping from the reference
        coordinate to the reference configuration.
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_kinematics_t1d1")

    dN_dxi = shape_function_derivatives_t1d1()
    xi_gp, _ = gauss_points_1d_two_point()
    J = he / 2.0  # dX/dxi in the reference configuration

    # B = du/dX = dN/dX = dN/dxi * dxi/dX
    B = dN_dxi / J

    F_gp = np.zeros_like(xi_gp)
    E_gp = np.zeros_like(xi_gp)

    # For a linear bar, du/dX and thus F are constant, but we keep the loop
    # so that the structure matches a general Gauss-point loop.
    du_dX = float(B @ ue)
    F = 1.0 + du_dX
    E = 0.5 * (F * F - 1.0)

    for a in range(xi_gp.size):
        F_gp[a] = F
        E_gp[a] = E

    return F_gp, E_gp, B, J


def element_internal_force_t1d1(
    xe: NDArray[np.float64],
    area: float,
    material: dict[str, Any],
    ue: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute internal nodal forces for a T1D1 element using Gauss quadrature.

    The internal nodal force vector is obtained by integrating

        f_int = integral of [B(xi)^T * sigma(xi) * A * J] over xi in [-1, 1],

    where sigma(xi) = E * strain(xi) and strain(xi) = B(xi) * ue.
    For the linear bar element the strain and stress are constant, but we
    still perform the integral using the same 2-point Gauss rule as in
    element_stiffness_t1d1 to mirror the standard finite element formulation.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape (2, num_dim).
    area:
        Cross-sectional area A of the bar.
    material:
        Material dictionary compatible with
        material_tangent_stiffness_elastic.
    ue:
        Element displacement vector with shape (2,).

    Returns
    -------
    ndarray
        Internal nodal force vector f_int with shape (2,).
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_internal_force_t1d1")

    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0

    fint = np.zeros(2, dtype=float)
    for a in range(xi_gp.size):
        B = dN_dxi / J
        strain = float(B @ ue)
        stress = material_stress_elastic(material, strain)
        # f_int contribution: B^T * sigma * A * J * w
        fint += B * stress * area * J * w_gp[a]

    return fint


def element_internal_force_neo_hooke_t1d1(
    xe: NDArray[np.float64],
    area: float,
    material: dict[str, Any],
    ue: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Internal force and tangent for a Neo-Hooke T1D1 bar element.

    The 1D Neo-Hooke model used in this project is

        sigma(F) = (E / 2) * (F - 1 / F),

    with stretch F = 1 + du/dX. For a two-node bar we approximate
    du/dX = B * u_e, where B is the strain–displacement vector and u_e
    are the element nodal displacements.

    The internal nodal force vector and consistent tangent stiffness are

        f_int = integral of [B^T * sigma(F) * A * J] over xi in [-1, 1]
        k_e   = integral of [B^T * (d sigma / dF) * B * A * J] over xi in [-1, 1]

    evaluated with 2-point Gauss quadrature.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape (2, num_dim).
    area:
        Cross-sectional area A.
    material:
        Material dictionary with parameters["E"] defining the modulus.
    ue:
        Element displacement vector with shape (2,).

    Returns
    -------
    (f_int, k_e)
        Internal nodal force vector and 2x2 consistent tangent matrix.
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_internal_force_neo_hooke_t1d1")

    # dN/dxi for 2-node bar; B = dN/dx = dN/dxi / J
    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0

    f_int = np.zeros(2, dtype=float)
    k_e = np.zeros((2, 2), dtype=float)

    for a in range(xi_gp.size):
        B = dN_dxi / J  # constant for the linear bar
        du_dX = float(B @ ue)
        F = 1.0 + du_dX

        P, dP_dF = material_neo_hooke_1d_pk1(material, F)

        # Internal force: B^T * P * A * J * w
        f_int += B * P * area * J * w_gp[a]

        # Tangent: B^T * (dP/dF) * B * A * J * w
        k_e += np.outer(B, B) * dP_dF * area * J * w_gp[a]

    return f_int, k_e


def element_residual_t1d1(
    f_int: NDArray[np.float64],
    f_ext: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the element residual vector r_e = f_int - f_ext.

    In the global Newton solve we assemble the residual from

        R(u) = f_int(u) - f_ext,

    so a converged solution satisfies R = 0 at all free degrees of freedom.
    """
    return f_int - f_ext


def element_external_force_t1d1(
    q: float,
    he: float,
) -> NDArray[np.float64]:
    """Compute equivalent nodal forces for a 1D line load using Gauss quadrature.

    The consistent nodal load vector for an axial line load q(x) is

        f_e = integral of [N(x)^T * q(x)] dx over the element.

    In this project we currently use a uniform load q = const, but the
    integral is evaluated with the same 2-point Gauss rule as the stiffness
    and internal force computations. For constant q this reproduces the
    familiar closed-form result

        f_e = q * he / 2 * [1, 1]^T.

    Parameters
    ----------
    q:
        Uniform line load q (force per unit length) in the axial direction.
        In future extensions this could be generalized to a position-dependent
        load evaluated at the Gauss points.
    he:
        Element length he.

    Returns
    -------
    ndarray
        Two-component nodal force vector for the element.
    """
    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0

    fe = np.zeros(2, dtype=float)
    for a in range(xi_gp.size):
        N = shape_functions_t1d1(xi_gp[a])
        # For now q is constant; later q(xi_gp[a]) could be used here instead.
        fe += N * q * J * w_gp[a]

    return fe


def element_stiffness_euler_bernoulli(
    E: float,
    I: float,
    L: float,
) -> NDArray[np.float64]:
    """Return the 4×4 Euler–Bernoulli beam stiffness matrix.

    Two-node beam with bending DOFs [w1, theta1, w2, theta2].
    E is Young's modulus, I is the second moment of area, and L is the
    element length.
    """
    if L <= 0.0:
        raise ValueError(f"Beam element length must be positive, got L = {L}")

    factor = E * I / (L**3)

    return factor * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
        ],
        dtype=float,
    )


def element_load_euler_bernoulli_uniform(
    q: float,
    L: float,
) -> NDArray[np.float64]:
    """Consistent nodal load vector for a uniform transverse load q.

    Two-node Euler–Bernoulli beam with DOFs [w1, theta1, w2, theta2].
    q is a constant load per unit length (positive in the chosen
    transverse direction), L is the element length.

    The result is [F1, M1, F2, M2] where F are equivalent nodal forces
    and M are equivalent nodal moments.
    """
    if L <= 0.0:
        raise ValueError(f"Beam element length must be positive, got L = {L}")

    return np.array(
        [
            q * L / 2.0,  # F1
            q * L * L / 12.0,  # M1
            q * L / 2.0,  # F2
            -q * L * L / 12.0,  # M2
        ],
        dtype=float,
    )


def element_load_euler_bernoulli_arbitrary(
    xe: NDArray[np.float64],
    q_func: Callable[[float], float],
) -> NDArray[np.float64]:
    """Consistent nodal load vector for arbitrary transverse load q(x).

    Two-node Euler–Bernoulli beam with DOFs [w1, theta1, w2, theta2].
    q_func(x) returns the transverse load at global coordinate x.

    This computes

        f_e = ∫_0^L N(x)^T q(x) dx

    using 2-point Gauss quadrature mapped from the reference coordinate
    xi ∈ [-1, 1]. The Hermite cubic shape functions are written in terms
    of the non-dimensional coordinate s = x / L ∈ [0, 1].
    """
    x1 = float(xe[0, 0])
    x2 = float(xe[1, 0])
    L = x2 - x1
    if L <= 0.0:
        raise ValueError(f"Beam element length must be positive, got L = {L}")

    # 2-point Gauss rule on xi ∈ [-1, 1]
    xi_gp, w_gp = gauss_points_1d_two_point()

    fe = np.zeros(4, dtype=float)

    for a in range(xi_gp.size):
        xi = float(xi_gp[a])
        w = float(w_gp[a])

        # Map xi ∈ [-1,1] → s ∈ [0,1] → x ∈ [x1,x2]
        s = 0.5 * (xi + 1.0)  # s = (xi + 1)/2
        x = x1 + s * L

        qx = float(q_func(x))

        # Hermite cubic shape functions in terms of s ∈ [0,1]
        N1 = 1.0 - 3.0 * s**2 + 2.0 * s**3  # w1
        N2 = L * (s - 2.0 * s**2 + s**3)  # theta1
        N3 = 3.0 * s**2 - 2.0 * s**3  # w2
        N4 = L * (-(s**2) + s**3)  # theta2

        N = np.array([N1, N2, N3, N4], dtype=float)

        # Jacobian: x(xi) ⇒ dx/dxi = L/2
        J = L / 2.0

        fe += N * qx * J * w

    return fe


def element_external_force_t1d1_arbitrary(
    xe: NDArray[np.float64],
    q_func: Callable[[float], float],
) -> NDArray[np.float64]:
    """Equivalent nodal force for arbitrary line load q(x).

    q_func(x) returns the load at physical coordinate x.
    """
    # Support both 1D coords (shape (2, 1)) and 2D coords (shape (2, 2)).
    x1 = float(xe[0, 0])
    x2 = float(xe[1, 0])
    L = x2 - x1

    # 2-point Gauss rule
    xi, w = gauss_points_1d_two_point()

    f = np.zeros(2, dtype=float)

    for i in range(2):
        xi_i = xi[i]
        w_i = w[i]

        # Shape functions
        N1 = 0.5 * (1 - xi_i)
        N2 = 0.5 * (1 + xi_i)

        # Map xi → x
        x = x1 * N1 + x2 * N2

        qx = q_func(x)

        # Jacobian
        J = L / 2.0

        # Consistent nodal load
        f[0] += N1 * qx * J * w_i
        f[1] += N2 * qx * J * w_i

    return f


# ---------------------------------------------------------------------------
# Global assembly helpers
# ---------------------------------------------------------------------------


def apply_neumann_bcs(
    F: NDArray[np.float64],
    bcs: list[dict[str, Any]],
    dof_per_node: int,
) -> None:
    """Add Neumann (force) boundary conditions to the global force vector.

    For each boundary-condition entry of type NEUMANN, this routine
    loops over the listed nodes, computes the corresponding global degree of
    freedom, and adds the prescribed value to the force vector F in place.
    """
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += float(bc["value"])


def apply_distributed_loads(
    F: NDArray[np.float64],
    dloads: list[dict[str, Any]],
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    block_elem_map: dict[int, tuple[int, int]],
    materials: dict[str, Any],
    dof_per_node: int,
) -> None:
    """Assemble equivalent nodal forces for all distributed loads."""
    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = float(np.sign(direction[0]))
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")

        profile = dload.get("profile", "UNIFORM").upper()

        for eid in dload["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dload.get('name', '<unnamed>')} not found in any element block"
                )

            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            he = float(xe[1, 0] - xe[0, 0])
            area = float(block["element"]["properties"]["area"])

            if dtype == "BX":
                # Body force per unit length, using profile
                if profile == "UNIFORM":
                    q = float(dload["value"]) * sign
                    qe = element_external_force_t1d1(q, he)

                elif profile == "EQUATION":
                    expr = dload["expression"]

                    def q_func(x, expr=expr, sign=sign):
                        # Simple environment for expression evaluation: only x and numpy
                        return float(eval(expr, {"x": x, "np": np})) * sign

                    qe = element_external_force_t1d1_arbitrary(xe, q_func)

                else:
                    raise ValueError(f"Unknown distributed-load profile {profile!r}")

            elif dtype == "GRAV":
                mat = materials[block["material"]]
                rho = float(mat.get("density", 0.0))
                q = rho * area * float(dload["value"]) * sign
                qe = element_external_force_t1d1(q, he)

            elif dtype == "QY":
                # Transverse load on Euler–Bernoulli beam.
                # Used with beam_fe_code (dof_per_node = 2).
                L = he  # beam length

                if profile == "UNIFORM":
                    q = float(dload["value"]) * sign
                    qe = element_load_euler_bernoulli_uniform(q, L)

                elif profile == "EQUATION":
                    expr = dload["expression"]

                    def q_func(x, expr=expr, sign=sign):
                        # Simple, controlled eval environment: x and numpy
                        return float(eval(expr, {"x": x, "np": np})) * sign

                    qe = element_load_euler_bernoulli_arbitrary(xe, q_func)

                else:
                    raise ValueError(f"Unknown distributed-load profile {profile!r}")

            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")

            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            F[eft] += qe


def collect_dirichlet_dofs(
    bcs: list[dict[str, Any]],
    dof_per_node: int,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Extract Dirichlet degrees of freedom and their prescribed values.

    This helper scans the boundary-condition list, selects entries of type
    DIRICHLET, and converts their node lists and local degrees of freedom
    into global degree-of-freedom indices.

    Returns
    -------
    tuple of (indices, values)
        indices is an integer array of global degree-of-freedom numbers and
        values is a float array of the corresponding prescribed
        displacement values.
    """
    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []

    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(float(bc["value"]))

    idx = np.array(prescribed_dofs, dtype=int)
    vals = np.array(prescribed_vals, dtype=float)
    return idx, vals


def solve_reduced_system(
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    prescribed_idx: NDArray[np.int_],
    prescribed_vals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the constrained linear system and assemble the full DOF vector.

    The global system K u = F is partitioned into free and prescribed
    degrees of freedom. The reduced system for the free part is

        K_ff * u_f = F_f - K_fp * u_p,

    where u_p contains the prescribed values. This routine

    1. builds the index sets of free and prescribed DOFs,
    2. forms K_ff and K_fp,
    3. solves for u_f, and
    4. assembles and returns the full displacement vector u.
    """
    num_dof = K.shape[0]
    all_dofs: NDArray[np.int_] = np.arange(num_dof, dtype=int)
    free_dofs: NDArray[np.int_] = np.setdiff1d(all_dofs, prescribed_idx)

    if free_dofs.size == 0:
        dofs = np.zeros(num_dof, dtype=float)
        dofs[prescribed_idx] = prescribed_vals
        return dofs

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_idx)]
    Ff = F[free_dofs] - Kfp @ prescribed_vals
    uf = np.linalg.solve(Kff, Ff)

    dofs = np.zeros(num_dof, dtype=float)
    dofs[free_dofs] = uf
    dofs[prescribed_idx] = prescribed_vals
    return dofs


# ---------------------------------------------------------------------------
# Global solver
# ---------------------------------------------------------------------------


def first_fe_code(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    bcs: list[dict[str, Any]],
    dloads: list[dict[str, Any]],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Assemble and solve a 1D linear-elastic finite element problem.

    This routine implements a small 1D bar solver for linear statics. It uses
    preprocessed data from wundy.ui and performs the following steps:

    1. Assemble the global stiffness matrix from all element blocks using
       element_stiffness_t1d1 and the material response functions.
    2. Assemble the global right-hand side from Neumann boundary conditions
       via apply_neumann_bcs and from distributed loads via
       apply_distributed_loads.
    3. Impose Dirichlet boundary conditions by collecting prescribed degrees
       of freedom with collect_dirichlet_dofs and solving the reduced
       linear system through solve_reduced_system.
    4. Return the full displacement vector, global stiffness matrix and
       assembled force vector.

    The problem is a linear, static, 1D bar with one translational degree of
    freedom per node and two-node T1D1 elements.

    Parameters
    ----------
    coords:
        Nodal coordinates with shape (num_node, num_dim); only the first
        column (x-coordinate) is used here.
    blocks:
        List of element-block dictionaries from wundy.ui.preprocess.
        Each block must provide the connectivity ("connect"), element
        description ("element") and material name ("material").
    bcs:
        Boundary-condition dictionaries describing Dirichlet and Neumann
        conditions in preprocessed form.
    dloads:
        Distributed-load dictionaries describing body forces of type
        "BX" (user-specified line load) or "GRAV" (gravity using
        material density).
    materials:
        Mapping from material name to dictionaries that contain at least a
        "parameters" sub-dictionary with the Young’s modulus "E" and,
        for gravitational loads, a "density" entry.
    block_elem_map:
        Mapping from global element index to (block_index, local_elem_index)
        specifying which block an element belongs to and its local index.

    Returns
    -------
    dict
        A dictionary with keys

        "dofs":
            Full vector of nodal displacements.
        "stiff":
            Global stiffness matrix K.
        "force":
            Global assembled force vector F.
    """
    dof_per_node: int = 1
    num_node = coords.shape[0]
    num_dof = int(num_node * dof_per_node)

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Element stiffness assembly
    for block in blocks:
        area = float(block["element"]["properties"]["area"])
        material = materials[block["material"]]
        E = material_tangent_stiffness_elastic(material)

        for nodes in block["connect"]:
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes]
            ke = element_stiffness_t1d1(xe, area, E)
            K[np.ix_(eft, eft)] += ke

    # External forces: Neumann + distributed loads
    apply_neumann_bcs(F, bcs, dof_per_node)
    apply_distributed_loads(F, dloads, coords, blocks, block_elem_map, materials, dof_per_node)

    # Dirichlet constraints and solution
    prescribed_idx, prescribed_vals = collect_dirichlet_dofs(bcs, dof_per_node)
    dofs = solve_reduced_system(K, F, prescribed_idx, prescribed_vals)

    solution: dict[str, Any] = {"dofs": dofs, "stiff": K, "force": F}
    return solution


def newton_bar_neo_hooke_1d(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    bcs: list[dict[str, Any]],
    dloads: list[dict[str, Any]],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
    max_iter: int = 25,
    tol: float = 1.0e-10,
) -> dict[str, Any]:
    """Solve a 1D bar problem with Neo-Hooke material using Newton–Raphson.

    This routine uses the same mesh, blocks, and load description as
    first_fe_code, but replaces the linear elastic constitutive
    model with the 1D Neo-Hooke law implemented in
    material_neo_hooke_1d_pk1. The global nonlinear system

        R(u) = f_int(u) - f_ext = 0

    is solved by Newton–Raphson on the free degrees of freedom using the
    consistent tangent stiffness assembled from
    element_internal_force_neo_hooke_t1d1.

    Parameters
    ----------
    coords, blocks, bcs, dloads, materials, block_elem_map:
        Same preprocessed data structures as used in first_fe_code.
    max_iter:
        Maximum number of Newton iterations.
    tol:
        Convergence tolerance on the infinity-norm of the residual at the
        free degrees of freedom.

    Returns
    -------
    dict
        Dictionary with keys

        "dofs":
            Converged nodal displacement vector.
        "stiff":
            Final global tangent stiffness matrix.
        "force_int":
            Global internal force vector at the final state.
        "force_ext":
            Global external force vector (Neumann + distributed loads).
        "residual":
            Final residual vector f_int - f_ext.
        "iterations":
            Number of Newton iterations performed.
    """
    dof_per_node: int = 1
    num_node = coords.shape[0]
    num_dof = int(num_node * dof_per_node)

    # External forces are independent of displacement -> assemble once.
    F_ext = np.zeros(num_dof, dtype=float)
    apply_neumann_bcs(F_ext, bcs, dof_per_node)
    apply_distributed_loads(F_ext, dloads, coords, blocks, block_elem_map, materials, dof_per_node)

    # Dirichlet data
    prescribed_idx, prescribed_vals = collect_dirichlet_dofs(bcs, dof_per_node)
    all_dofs: NDArray[np.int_] = np.arange(num_dof, dtype=int)
    free_dofs: NDArray[np.int_] = np.setdiff1d(all_dofs, prescribed_idx)

    # Initial guess: zero displacement, with prescribed DOFs set directly
    u = np.zeros(num_dof, dtype=float)
    if prescribed_idx.size:
        u[prescribed_idx] = prescribed_vals

    for it in range(max_iter):
        K = np.zeros((num_dof, num_dof), dtype=float)
        F_int = np.zeros(num_dof, dtype=float)

        # --- Assemble internal forces and tangent stiffness ---
        for block in blocks:
            area = float(block["element"]["properties"]["area"])
            material = materials[block["material"]]

            for nodes in block["connect"]:
                eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
                xe = coords[nodes]
                ue = u[eft]

                f_int_e, k_e = element_internal_force_neo_hooke_t1d1(xe, area, material, ue)

                F_int[eft] += f_int_e
                K[np.ix_(eft, eft)] += k_e

        # --- Global residual and convergence check ---
        R = F_int - F_ext
        R_free = R[free_dofs]
        res_norm = float(np.linalg.norm(R_free, ord=np.inf))

        if res_norm < tol:
            return {
                "dofs": u,
                "stiff": K,
                "force_int": F_int,
                "force_ext": F_ext,
                "residual": R,
                "iterations": it,
            }

        # --- Newton update on free DOFs ---
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        du_f = -np.linalg.solve(K_ff, R_free)
        u[free_dofs] += du_f

    raise RuntimeError(
        f"Newton solver did not converge in {max_iter} iterations; final residual norm = {res_norm}"
    )


def beam_fe_code(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    bcs: list[dict[str, Any]],
    dloads: list[dict[str, Any]],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Assemble and solve a 1D Euler–Bernoulli beam problem.

    This solver assumes:
      * two degrees of freedom per node: w (transverse displacement) and
        theta (rotation),
      * two-node beam elements,
      * uniform transverse distributed loads (type 'QY') using
        element_load_euler_bernoulli_uniform.

    The YAML/preprocess path is the same as for first_fe_code; the
    difference is only in how element stiffness and loads are formed.
    """
    dof_per_node: int = 2
    num_node = coords.shape[0]
    num_dof = int(num_node * dof_per_node)

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # --- Assemble element stiffness matrices ---
    for block in blocks:
        # Material: we reuse E from the existing elastic material
        material = materials[block["material"]]
        E = material_tangent_stiffness_elastic(material)

        # Element properties: need second moment of area I
        props = block["element"]["properties"]
        try:
            I = float(props["I"])
        except KeyError as exc:
            raise KeyError(
                "Beam solver expects element property 'I' "
                "(second moment of area) in the YAML element block."
            ) from exc

        for nodes in block["connect"]:
            # Beam element length from nodal coordinates (x only)
            xe = coords[nodes]
            L = float(xe[1, 0] - xe[0, 0])
            if L <= 0.0:
                raise ValueError(f"Beam element length must be positive, got L = {L}")

            ke = element_stiffness_euler_bernoulli(E, I, L)

            # DOF ordering per element: [w1, theta1, w2, theta2]
            eft = [
                global_dof(nodes[0], 0, dof_per_node),  # w1
                global_dof(nodes[0], 1, dof_per_node),  # theta1
                global_dof(nodes[1], 0, dof_per_node),  # w2
                global_dof(nodes[1], 1, dof_per_node),  # theta2
            ]
            K[np.ix_(eft, eft)] += ke

    # --- External forces: point loads + distributed loads ---
    apply_neumann_bcs(F, bcs, dof_per_node)
    apply_distributed_loads(F, dloads, coords, blocks, block_elem_map, materials, dof_per_node)

    # --- Dirichlet boundary conditions and solve ---
    prescribed_idx, prescribed_vals = collect_dirichlet_dofs(bcs, dof_per_node)
    dofs = solve_reduced_system(K, F, prescribed_idx, prescribed_vals)

    return {"dofs": dofs, "stiff": K, "force": F}


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree-of-freedom index for a node and local DOF."""
    return node * dof_per_node + local_dof
