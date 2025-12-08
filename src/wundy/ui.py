import logging
from typing import IO, Any, Dict

import numpy as np
import yaml

from .schemas import NEUMANN, DIRICHLET, input_schema
from .solver import newton_solve_1d, SolverOptions

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_dof(dof):
    """Convert dof from ('x','X',0) → integer 0."""
    if isinstance(dof, int):
        return dof
    if isinstance(dof, str):
        if dof.lower() == "x":
            return 0
    raise ValueError(f"Unsupported dof: {dof}")


def unique_name(existing_names: set, stem: str) -> str:
    """Generate unique sequential names: STEM-1, STEM-2…"""
    i = 1
    while True:
        name = f"{stem.upper()}-{i}"
        if name not in existing_names:
            existing_names.add(name)
            return name
        i += 1


def set_element_defaults(elem: dict[str, Any]) -> bool:
    """Defaults for T1D1."""
    if elem["type"].upper() == "T1D1":
        nft = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        elem.setdefault("properties", {})
        elem["properties"].update({
            "area": elem["properties"].get("area", 1.0),
            "node_per_elem": 2,
            "freedom_table": [nft, nft],
        })
        return True
    raise ValueError(f"Unknown element type {elem['type']}")


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load(file: IO[Any] | str) -> dict[str, dict[str, Any]]:
    if isinstance(file, str):
        with open(file, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(file)
    return input_schema.validate(data)


# ----------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------
def preprocess(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inp = data["wundy"]
    pre = {}
    errors = 0

    # -------------------- nodes --------------------
    num_node = len(inp["nodes"])
    max_dim = max(len(n[1:]) for n in inp["nodes"])
    node_map = pre.setdefault("node_map", {})
    coords = pre["coords"] = np.zeros((num_node, max_dim))

    for i, node in enumerate(inp["nodes"]):
        nid, *xc = node
        node_map[nid] = i
        coords[i, :len(xc)] = xc

    # -------------------- elements --------------------
    elem_map = pre.setdefault("elem_map", {})
    for i, element in enumerate(inp["elements"]):
        elem_map[element[0]] = i

    # -------------------- node sets --------------------
    nsets = pre.setdefault("nsets", {})
    nsets["ALL"] = list(range(num_node))

    for ns in inp.get("node sets", []):
        name = ns["name"]
        mapped = []
        for n in ns["nodes"]:
            mapped.append(node_map[n])
        nsets[name] = mapped

    # -------------------- element sets --------------------
    elsets = pre.setdefault("elsets", {})
    elsets["ALL"] = list(range(len(inp["elements"])))

    for es in inp.get("element sets", []):
        name = es["name"]
        mapped = []
        for e in es["elements"]:
            mapped.append(elem_map[e])
        elsets[name] = mapped

    # -------------------- materials --------------------
    materials = pre.setdefault("materials", {})
    for m in inp["materials"]:
        materials[m["name"].upper()] = {
            "type": m["type"].upper(),
            "parameters": m["parameters"],
        }

    # -------------------- blocks --------------------
    blocks = pre.setdefault("blocks", [])
    for eb in inp["element blocks"]:
        bname = eb["name"].upper()
        mat = eb["material"].upper()

        block = {
            "name": bname,
            "material": mat,
        }

        elem_spec = eb["elements"]
        if isinstance(elem_spec, str):
            elems = elsets[elem_spec]
        else:
            elems = [elem_map[e] for e in elem_spec]

        econnect = []
        local_map = {}
        for idx, e in enumerate(elems):
            _, n1, n2 = inp["elements"][e]
            econnect.append([node_map[n1], node_map[n2]])
            local_map[e] = idx

        block["connect"] = np.array(econnect, dtype=int)
        block["elem_map"] = local_map

        block["element"] = eb["element"]
        set_element_defaults(block["element"])

        blocks.append(block)

    # -------------------- BCs + concentrated loads --------------------
    bcs = pre.setdefault("bcs", [])
    used_names = set()

    # BCs
    for bc in inp["boundary conditions"]:
        name = bc.get("name")
        if name is None:
            name = unique_name(used_names, "BOUNDARY")
        else:
            name = name.upper()
            used_names.add(name)

        nodes = []
        if isinstance(bc["nodes"], str):
            nodes = nsets[bc["nodes"]]
        else:
            nodes = [node_map[n] for n in bc["nodes"]]

        local_dof = _normalize_dof(bc["dof"])

        bcs.append({
            "name": name,
            "local_dof": local_dof,
            "type": DIRICHLET,
            "nodes": nodes,
            "value": bc.get("value", 0.0),
        })

    # Loads
    for cl in inp.get("concentrated loads", []):
        name = cl.get("name")
        if name is None:
            name = unique_name(used_names, "CLOAD")
        else:
            name = name.upper()
            used_names.add(name)

        nodes = []
        if isinstance(cl["nodes"], str):
            nodes = nsets[cl["nodes"]]
        else:
            nodes = [node_map[n] for n in cl["nodes"]]

        local_dof = _normalize_dof(cl.get("dof", "x"))

        bcs.append({
            "name": name,
            "local_dof": local_dof,
            "type": NEUMANN,
            "nodes": nodes,
            "value": cl["value"],
        })

    # -------------------- distributed loads --------------------
    dload = pre.setdefault("dload", [])
    for dl in inp.get("distributed loads", []):
        elems = dl["elements"]
        if isinstance(elems, str):
            elems = elsets[elems]
        else:
            elems = [elem_map[e] for e in elems]

        dload.append({
            "name": dl.get("name", "").upper(),
            "elements": elems,
            "type": dl["type"],
            "value": dl["value"],
            "direction": dl["direction"],
        })

    # -------------------- block_elem_map (tuple version) --------------------
    bem = pre.setdefault("block_elem_map", {})
    for block in blocks:
        lst = []
        for ge, le in block["elem_map"].items():
            n1, n2 = block["connect"][le]
            lst.append((ge, n1, n2))
        bem[block["name"]] = lst

    return pre


# ----------------------------------------------------------------------
# Solve wrapper
# ----------------------------------------------------------------------
def solve(filename: str, options: dict[str, Any] | None = None):
    with open(filename, "r") as f:
        data = load(f)

    pre = preprocess(data)
    solver_opts = SolverOptions(**options) if options else SolverOptions()
    return newton_solve_1d(pre, options=solver_opts)
