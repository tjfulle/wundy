import logging
from typing import IO
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .block import Block
from .constants import DIRICHLET
from .constants import NEUMANN
from .element import Element
from .material import Material
from .solver import Solver
from .ui import load

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, nodes: list[list[int | float]], elements: list[list[int]]) -> None:
        self.prepared = False
        num_node: int = len(nodes)
        max_dim: int = max(len(n[1:]) for n in nodes)
        self.coords: NDArray[float] = np.zeros((num_node, max_dim), dtype=float)
        self.node_map: dict[int, int] = {}
        for i, node in enumerate(nodes):
            nid, *xc = node
            self.node_map[int(nid)] = i
            self.coords[i, : len(xc)] = [float(x) for x in xc]

        self.node_sets: dict[str, list[int]] = {}
        self.node_sets["ALL"] = list(range(num_node))

        num_elem: int = len(elements)
        max_elem: int = max(len(e[1:]) for e in elements)
        self.connect: NDArray[int] = -np.ones((num_elem, max_elem), dtype=int)
        self.elem_map: dict[int, int] = {}
        num_elem: int = len(elements)
        for i, element in enumerate(elements):
            self.elem_map[int(element[0])] = i
            for j, node in enumerate(element[1:]):
                if node not in self.node_map:
                    raise ValueError(f"Node {j + 1} of element {i + 1} ({node}) is not defined")
                self.connect[i, j] = self.node_map[node]

        self.element_sets: dict[str, list[int]] = {}
        self.element_sets["ALL"] = list(range(num_elem))

        self.materials: dict[str, Material] = {}
        self.blocks: list[Block] = []
        self.boundary: list[dict] = []
        self.distributed_loads: list[dict] = []
        self.equations: list[list[tuple[int, int, float]]] = []
        self._solver: Solver | None = None

        # dof_map[node, dof] -> global (model) dof
        self.dof_map: NDArray[int] = np.empty((0, 0), dtype=int)
        self.node_signatures: NDArray[int] = np.empty((0, 0), dtype=int)

        # block, block dof -> model dof
        self.block_dof_map: NDArray[int] = np.empty((0, 0), dtype=int)

        # block, block element -> model element
        self.block_ele_map: NDArray[int] = np.empty((0, 0), dtype=int)

        # block, block node -> model node
        self.block_nod_map: NDArray[int] = np.empty((0, 0), dtype=int)

        self.dirichlet_dofs: NDArray[int] = np.array(0, dtype=int)
        self.dirichlet_vals: NDArray[float] = np.array(0, dtype=float)
        self.neumann_dofs: NDArray[int] = np.array(0, dtype=int)
        self.neumann_vals: NDArray[float] = np.array(0, dtype=float)

        self.solution: dict[str, Any] = {}

    @classmethod
    def from_file(cls, file: str | IO[Any]) -> "Model":
        fown: bool = isinstance(file, str)
        fh: IO[Any] = file if hasattr(file, "read") else open(file)  # ty: ignore[invalid-assignment]
        try:
            data = load(fh)
        finally:
            if fown:
                fh.close()
        inp = data["wundy"]
        self = cls(inp["nodes"], inp["elements"])

        if node_sets := inp.get("node sets"):
            for ns in node_sets:
                self.add_node_set(ns["name"], ns["nodes"])
        if element_sets := inp.get("element sets"):
            for es in element_sets:
                self.add_element_set(es["name"], es["elements"])
        for m in inp["materials"]:
            self.add_material(m["name"], m["type"], m["parameters"])
        for b in inp["element blocks"]:
            self.add_element_block(b["name"], b["material"], b["element"], b["elements"])
        for bc in inp["boundary conditions"]:
            if not bc.get("name"):
                stem = "ConcentratedLoad" if bc["type"] == NEUMANN else "Bounary"
                bc["name"] = unique_name(inp["boundary conditions"], stem)
            if bc["type"] == DIRICHLET:
                self.add_dirichlet_boundary(bc["name"], bc["nodes"], bc["dof"], bc["value"])
            elif bc["type"] == NEUMANN:
                self.add_concentrated_load(bc["name"], bc["nodes"], bc["dof"], bc["value"])
        for cl in inp.get("concentrated loads", []):
            if not cl.get("name"):
                cl["name"] = unique_name(inp["concentrated loads"], "ConcentratedLoad")
            self.add_concentrated_load(cl["name"], cl["nodes"], cl["dof"], cl["value"])
        for dl in inp.get("distributed loads", []):
            if not dl.get("name"):
                dl["name"] = unique_name(inp["distributed loads"], "DistributedLoad")
            self.add_distributed_load(
                dl["name"], dl["type"], dl["elements"], dl["value"], dl["direction"]
            )
        for eq in inp.get("equations", []):
            self.add_equation(eq)
        if s := inp.get("solver"):
            options = s.get("options", {})
            method = options.pop("method", None)
            if s["type"] == "NONLINEAR" and method is None:
                method = "NEWTON"
            self.set_solver(s["type"], method, **options)
        return self

    @property
    def solver(self) -> Solver:
        if self._solver is None:
            self._solver = Solver.factory("DIRECT")
        assert self._solver is not None
        return self._solver

    def set_solver(self, type: str, method: str | None = None, **options: Any) -> None:
        self._solver = Solver.factory(type, method, **options)

    @solver.setter
    def solver(self, arg: Solver) -> None:
        self._solver = arg

    def solve(self) -> None:
        if not self.prepared:
            self.prepare()
        self.solution.clear()
        solution = self.solver(self)
        self.solution.update(solution)

    def add_node_set(self, name: str, nodes: list[int]) -> None:
        if name in self.node_sets:
            raise ValueError(f"Duplicate node set {name!r}")
        ns = self.node_sets.setdefault(name, [])
        for node in nodes:
            if node not in self.node_map:
                raise ValueError(f"Node {node} in node set {name} is not defined")
            ns.append(self.node_map[node])

    def add_element_set(self, name: str, elements: list[int]) -> None:
        if name in self.element_sets:
            raise ValueError(f"Duplicate element set {name!r}")
        es = self.element_sets.setdefault(name, [])
        for e in elements:
            if e not in self.elem_map:
                raise ValueError(f"Element {e} in element set {name} is not defined")
            es.append(self.elem_map[e])

    def add_material(self, name: str, type: str, parameters: dict[str, float]) -> None:
        # Put materials in dictionary for easier look up
        if name in self.materials:
            raise ValueError(f"Duplicate material {name!r}")
        self.materials[name] = Material.factory(type, parameters)

    def add_dirichlet_boundary(
        self, name: str, nodes: str | list[int], dof: int, value: float
    ) -> None:
        ns: list[int] = []
        if isinstance(nodes, str):
            if nodes not in self.node_sets:
                raise ValueError(
                    f"Nodeset {nodes}, required by boundary condition {name}, is not defined"
                )
            ns.extend(self.node_sets[nodes])
        else:
            for n in nodes:
                if n not in self.node_map:
                    raise ValueError(
                        f"Node {n}, required by boundary condition {name}, is not defined"
                    )
                ns.append(self.node_map[n])
        if dof != 1:
            # FIXME: Relax this restriction when additional DOFs are added
            raise ValueError(f"DOF {dof}, required by boundary condition {name}, must be 1")

        self.boundary.append(
            {
                "name": name,
                "local_dof": dof - 1,
                "type": DIRICHLET,
                "nodes": ns,
                "value": value,
            }
        )

    def add_element_block(
        self, name: str, material: str, elemspec: dict[str, Any], elements: str | list[int]
    ) -> None:
        if name in [b.name for b in self.blocks]:
            raise ValueError(f"Duplicate element block {name!r}")

        if material not in self.materials:
            raise ValueError(
                f"material {material!r}, required by element block {name}, not defined"
            )

        if isinstance(elements, str):
            # elements given as set name
            if elements not in self.element_sets:
                raise ValueError(
                    f"Element set {elements!r}, required by element block {name}, not defined"
                )
            elements = self.element_sets[elements]
        else:
            for i, e in enumerate(elements):
                if e not in self.elem_map:
                    raise ValueError(
                        f"Element {e}, required for element block {name}, is not defined"
                    )
                elements[i] = self.elem_map[e]

        if not elements:
            raise ValueError(f"No elements defined for element block {name}")

        # By this point, elems holds the model element ID corresponding to the user element ID
        # passed in by ``elements``
        elems = np.asarray(elements, dtype=int)
        assigned_elements = self.block_ele_map[np.where(self.block_ele_map != -1)]
        mask = np.isin(elems, assigned_elements)
        if np.any(mask):
            duplicates = ", ".join(str(e) for e in elems[mask])
            raise ValueError(
                f"Block {name}: attempting to assign elements {duplicates} "
                "which are already assigned to other element blocks"
            )

        element = Element.factory(elemspec["type"], elemspec["properties"])
        nodes_per_elem = len(element.node_freedom_table)
        connect = self.connect[elems, :nodes_per_elem]
        if np.any(connect == -1):
            raise ValueError(
                f"Block {name}: wrong number of nodes in one or more elements "
                f"(elements in this block must have {nodes_per_elem} nodes"
            )

        # E = self.block_ele_map[b, e] is the model (internal) element number "E" for the
        # eth internal element of block b
        if not self.block_ele_map.size:
            self.block_ele_map = elems.reshape((1, -1))
        else:
            self.block_ele_map = hstack(self.block_ele_map, elems)

        # Connect contains the node IDs for the block connectivity in the model system
        # Convert to local block IDs
        nodes, inverse = np.unique(connect, return_inverse=True)
        xb = self.coords[nodes]
        cb = inverse.reshape(connect.shape)
        block = Block(name, xb, cb, element, self.materials[material])
        self.blocks.append(block)

        # N = self.block_nod_map[b, n] is the model (internal) node number "N" for the
        # nth internal node of block b
        if not self.block_nod_map.size:
            self.block_nod_map = nodes.reshape((1, -1))
        else:
            self.block_nod_map = hstack(self.block_nod_map, nodes)

    def reverse_lookup(self, **kwds: int) -> int:
        if len(kwds) > 1:
            raise ValueError("Can only reverse look up one item at a time")
        if element := kwds.get("element"):
            for ue, ge in self.elem_map.items():
                if ge == element:
                    return ue
            raise ValueError(f"Cannot find user element mapped to global element {element}")
        elif node := kwds.get("node"):
            for un, gn in self.node_map.items():
                if gn == node:
                    return un
            raise ValueError(f"Cannot find user node mapped to global node {node}")
        raise ValueError("Missing required keyword argument")

    def add_concentrated_load(
        self, name: str, nodes: str | list[int], dof: int, value: float
    ) -> None:
        ns: list[int] = []
        if isinstance(nodes, str):
            if nodes not in self.node_sets:
                raise ValueError(
                    f"Nodeset {nodes}, required by boundary condition {name}, is not defined"
                )
            ns.extend(self.node_sets[nodes])
        else:
            for n in nodes:
                if n not in self.node_map:
                    raise ValueError(
                        f"Node {n}, required by boundary condition {name}, is not defined"
                    )
                ns.append(self.node_map[n])
        if dof != 1:
            # FIXME: Relax this restriction when additional DOFs are added
            raise ValueError(f"DOF {dof}, required by boundary condition {name}, must be 1")

        self.boundary.append(
            {
                "name": name,
                "local_dof": dof - 1,
                "type": NEUMANN,
                "nodes": ns,
                "value": value,
            }
        )

    def add_distributed_load(
        self, name: str, type: str, elements: str | list[int], value: float, direction: list[float]
    ) -> None:
        # Process distributed load
        if name in [_["name"] for _ in self.distributed_loads]:
            raise ValueError(f"Duplicate distributed load name {name!r}")
        elems: list[int] = []
        if isinstance(elements, str):
            # elements given as set name
            if elements not in self.element_sets:
                raise ValueError(
                    f"Element set {elements!r}, required by distributed load {name}, not defined"
                )
            elems.extend(self.element_sets[elements])
        else:
            for e in elements:
                if e not in self.elem_map:
                    raise ValueError(
                        f"Element {e}, required by distributed load {name}, is not defined"
                    )
                elems.append(self.elem_map[e])
        if len(direction) != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")
        self.distributed_loads.append(
            {
                "name": name,
                "elements": np.asarray(elems, dtype=int),
                "addr": None,  # can't know until prepare() is called
                "type": type,
                "value": value,
                "direction": np.asarray(direction, dtype=float),
            }
        )

    def add_equation(self, equation: list[tuple[int, int, float]]) -> None:
        # Process multi-point constraint equations
        eq: list[tuple[int, int, float]] = []
        for node, dof, coeff in equation:
            if node not in self.node_map:
                raise ValueError(f"Node {node} is not defined")
            if dof != 1:
                raise ValueError(f"{dof=} not implemented (choose dof=1)")
            eq.append((self.node_map[node], dof - 1, coeff))
        self.equations.append(eq)

    def prepare(self) -> None:
        # Check if all elements are assigned to an element block
        assigned_elements = self.block_ele_map[np.where(self.block_ele_map != -1)]
        num_elem = len(self.elem_map)
        if unassigned := set(range(num_elem)).difference(assigned_elements):
            s = ", ".join(str(_) for _ in unassigned)
            raise ValueError(f"Elements {s} not assigned to any element blocks")

        # Check that all nodes belong to an element or have a Dirichlet BC
        connected: set[int] = set([_ for _ in self.connect.flatten() if _ != -1])
        allnodes: set[int] = set(range(self.coords.shape[0]))
        if disconnected := allnodes.difference(connected):
            for node in disconnected:
                for bc in self.boundary:
                    if bc["type"] == DIRICHLET and node in bc["nodes"]:
                        break
                else:
                    n = self.reverse_lookup(node=node)
                    raise ValueError(
                        f"Node {n} is not connected to any element "
                        "and does not have an associated dirichlet BC."
                    )

        # Build node signatures and dof maps
        num_node: int = self.coords.shape[0]
        max_dof_per_node = max([b.element.dof_per_node for b in self.blocks])
        # FIX ME: compress out dofs not used at all
        max_dof_per_node = 10
        self.node_signatures = np.zeros((num_node, max_dof_per_node), dtype=int)
        for i, block in enumerate(self.blocks):
            node_freedoms = np.asarray(block.element.node_freedom_table, dtype=int)
            for nodes in block.connect:
                # Map block local node number to model node number
                ix = [self.block_nod_map[i, node] for node in nodes]
                self.node_signatures[ix, :] |= node_freedoms

        # Check for dummy nodes that are not included in element connectivity
        # All dummy nodes must have associated Dirichlet BCs.  We can use the BC to fill in the node
        # signature
        for bc in self.boundary:
            self.node_signatures[bc["nodes"], bc["local_dof"]] |= 1

        if np.any(self.node_signatures.sum(axis=1) == 0):
            missing = np.where(self.node_signatures.sum(axis=1) == 0)[0]
            raise ValueError(f"Dummy node without Dirichlet BC: {missing.tolist()}")

        active_mask: NDArray[bool] = self.node_signatures == 1
        self.num_dof: int = int(active_mask.sum())

        mask = active_mask.ravel()
        map = -np.ones_like(mask, dtype=int)
        map[mask] = np.arange(self.num_dof)
        self.dof_map = map.reshape(self.node_signatures.shape).copy()

        # Create map
        # DOF = block_dof_map[b, dof] is the global model DOF for dof of block b
        m = max([block.num_dof for block in self.blocks])
        self.block_dof_map = -np.ones((len(self.blocks), m), dtype=int)
        for i, block in enumerate(self.blocks):
            nodes = np.unique(block.connect)
            for node in nodes:
                N = self.block_nod_map[i, node]
                for local_dof in range(block.element.dof_per_node):
                    dof = block.dof_map[node, local_dof]
                    DOF = self.dof_map[N, local_dof]
                    self.block_dof_map[i, dof] = DOF

        # extract Dirichlet and Neumann BCs
        dirichlet_dofs: list[int] = []
        dirichlet_vals: list[float] = []
        neumann_dofs: list[int] = []
        neumann_vals: list[float] = []
        for bc in self.boundary:
            if bc["type"] == DIRICHLET:
                for n in bc["nodes"]:
                    I = self.dof_map[n, bc["local_dof"]]
                    dirichlet_dofs.append(I)
                    dirichlet_vals.append(bc["value"])
            elif bc["type"] == NEUMANN:
                for n in bc["nodes"]:
                    I = self.dof_map[n, bc["local_dof"]]
                    neumann_dofs.append(I)
                    neumann_vals.append(bc["value"])

        if dirichlet_dofs:
            self.dirichlet_dofs = np.asarray(dirichlet_dofs)
            self.dirichlet_vals = np.asarray(dirichlet_vals)

        if neumann_dofs:
            self.neumann_dofs = np.asarray(neumann_dofs)
            self.neumann_vals = np.asarray(neumann_vals)

        # Find block and local block element ID for elements participating in distributed loads
        for dload in self.distributed_loads:
            addr: list[tuple[int, int]] = []
            for i, eid in enumerate(dload["elements"]):
                b, e = self.find_block_elem(eid)
                addr.append((int(b), int(e)))
            dload["addr"] = addr

        self.prepared = True

    def assemble_system(
        self, u: NDArray[float], du: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float]]:
        if not self.prepared:
            raise RuntimeError("Model is not in prepared state")
        K = np.zeros((self.num_dof, self.num_dof), dtype=float)
        F = np.zeros(self.num_dof, dtype=float)
        for i, block in enumerate(self.blocks):
            mask = self.block_mask(i)
            ix = self.block_dof_map[i][mask]
            kb, fb = block.assemble(u[ix], du[ix])
            bft = np.arange(self.num_dof, dtype=int)[ix]
            K[np.ix_(bft, bft)] += kb
            F[np.ix_(bft)] += fb
        return K, F

    def block_mask(self, blockno: int) -> NDArray[bool]:
        """Returns a boolean array that is True where the block's dofs are active"""
        return np.where(self.block_dof_map[blockno] != -1)[0]

    def global_force(self, u: NDArray[float], du: NDArray[float]) -> NDArray[float]:
        F = np.zeros(self.num_dof, dtype=float)
        self.apply_neumann_bcs(F)
        self.apply_distributed_loads(F)
        return F

    def apply_neumann_bcs(self, F: NDArray[float]) -> None:
        # Apply Neumann boundary conditions to force
        if self.neumann_dofs.size:
            F[self.neumann_dofs] += self.neumann_vals

    def apply_dirichlet_bcs(
        self,
        K: NDArray[float],
        F: NDArray[float],
        u: NDArray[float],
        du: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        Kbc, Fbc = K.copy(), F.copy()
        ubc = np.zeros_like(self.dirichlet_vals)
        for i, dof in enumerate(self.dirichlet_dofs):
            ubc[i] = self.dirichlet_vals[i] - u[dof]
            Fbc -= K[:, dof] * ubc[i]
            Kbc[:, dof] = Kbc[dof, :] = 0.0
            Kbc[dof, dof] = 1.0
        Fbc[self.dirichlet_dofs] = ubc
        return Kbc, Fbc

    def apply_distributed_loads(self, F: NDArray[float]) -> None:
        # Apply distributed loads
        for dload in self.distributed_loads:
            dtype = dload["type"]
            direction = np.array(dload["direction"], dtype=float)
            sign = np.sign(direction[0])
            for b, e in dload["addr"]:
                block = self.blocks[b]
                ix = block.connect[e]
                xe = block.coords[ix]
                if dtype == "BX":
                    q = dload["value"] * sign
                elif dtype == "GRAV":
                    A = block.element.area(xe)
                    rho = block.material.density
                    q = rho * A * dload["value"] * sign
                else:
                    raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")
                fe = block.element.force(xe, q)
                nodes = self.block_nod_map[b, ix]
                nft = block.element.node_freedom_table
                eft = self.element_freedom_table(nodes, nft)
                F[eft] += fe

    def apply_linear_constraints(
        self,
        K: NDArray[float],
        F: NDArray[float],
        u: NDArray[float],
        du: NDArray[float],
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
        assert len(self.equations) > 0
        # Build the linear constrain matrix
        m = len(self.equations)
        n = K.shape[0]
        C: NDArray[float] = np.zeros_like(K, shape=(m, n))
        for i, equation in enumerate(self.equations):
            for node, dof, coeff in equation:
                I = self.dof_map[node, dof]
                C[i, I] = coeff

        # Gather prescribed DOFs and values
        r: NDArray[float] = np.zeros(m, dtype=float)
        for i, row in enumerate(C):
            for j, dof in enumerate(self.dirichlet_dofs):
                coeff = row[dof]
                if abs(coeff) > 0.0:
                    r[i] -= coeff * (self.dirichlet_vals[j] - u[dof])
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

    def element_freedom_table(
        self, nodes: list[int], node_freedoms: list[tuple[int, ...]]
    ) -> list[int]:
        eft = [
            self.dof_map[node, j]
            for n, node in enumerate(nodes)
            for j, active in enumerate(node_freedoms[n])
            if active
        ]
        return eft

    def find_block_elem(self, element: int) -> tuple[int, int]:
        # Find the block index and local block element corresponding to element
        rows, cols = np.where(self.block_ele_map == element)
        return rows[0], cols[0]


def unique_name(named_items: list[dict], stem: str) -> str:
    names = [item.get("name") for item in named_items]
    i = 1
    while True:
        name = f"{stem}-{i}"
        if name not in names:
            return name
        i += 1


def hstack(rows: NDArray, row: NDArray) -> NDArray:
    dtype = rows.dtype
    if row.size > rows.shape[1]:
        # add new colums to rows to make it have equal columns as row
        shape = (rows.shape[0], row.size - rows.shape[1])
        rows = np.hstack((rows, -np.ones(shape, dtype=dtype)), dtype=dtype)
    elif row.size < rows.shape[1]:
        # add new colums to row to make it have equal columns as rows
        size = rows.shape[1] - row.size
        row = np.append(row, -np.ones(size, dtype=dtype))
    assert row.size == rows.shape[1]
    return np.vstack((rows, row), dtype=dtype)
