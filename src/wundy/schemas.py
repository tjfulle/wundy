from typing import Any

from schema import And, Or, Optional, Use, Schema

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
NEUMANN = 0
DIRICHLET = 1

# Only 1D bar elements for this project
element_types = {"T1D1"}
bc_types = {"DIRICHLET", "NEUMANN"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_case(s: str) -> str:
    return s.upper()


def valid_element_type(name: str) -> bool:
    return normalize_case(name) in element_types


def isnumeric(x) -> bool:
    return isinstance(x, (int, float))


def ispositive(x) -> bool:
    return x > 0


def list_of_int(seq) -> bool:
    return all(isinstance(n, int) for n in seq)


def list_of_list(seq) -> bool:
    return all(isinstance(n, list) for n in seq)


# ---------------------------------------------------------------------
# DOF handling – ONLY X (mapped to 0)
# ---------------------------------------------------------------------
def valid_dof_id(dof: str) -> bool:
    return normalize_case(dof) == "X"


def dof_id_to_enum(dof: str) -> int:
    return 0  # only one DOF in this project


# ---------------------------------------------------------------------
# Material validation
# ---------------------------------------------------------------------
def validate_material_parameters(material: dict[str, Any]) -> bool:
    params = material["parameters"]
    if "E" not in params:
        raise ValueError("Material must define E")

    E = params["E"]
    if not isnumeric(E) or E <= 0:
        raise ValueError("E must be > 0")

    # nu is optional; if present, check range
    if "nu" in params:
        nu = params["nu"]
        if not isnumeric(nu) or not (-1.0 <= nu < 0.5):
            raise ValueError("nu must be between -1 and 0.5")

    # Allow both ELASTIC and NEO_HOOKE in this project
    mtype = normalize_case(material["type"])
    if mtype not in {"ELASTIC", "NEO_HOOKE"}:
        raise ValueError(f"Unknown material type {material['type']!r}")

    return True


material_schema = Schema(
    And(
        {
            "type": And(str, Use(normalize_case)),
            "name": And(str, Use(normalize_case)),
            "parameters": {str: object},
            Optional("density", default=0.0): And(isnumeric, ispositive),
        },
        validate_material_parameters,
    )
)


# ---------------------------------------------------------------------
# Nodes: [[id, x], ...]
# ---------------------------------------------------------------------
nodes_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(isinstance(inner[0], int) for inner in outer),
        lambda outer: all(
            isinstance(f, (int, float)) for inner in outer for f in inner[1:]
        ),
        Use(
            lambda outer: [
                [int(inner[0]), *[float(_) for _ in inner[1:]]] for inner in outer
            ]
        ),
    )
)


# ---------------------------------------------------------------------
# Elements: [[id, n1, n2], ...]
# ---------------------------------------------------------------------
elements_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(list_of_int(inner) for inner in outer),
    )
)


# ---------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------
nset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "nodes": And(list, list_of_int),
    }
)

elset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "elements": And(list, list_of_int),
    }
)


# ---------------------------------------------------------------------
# Boundary conditions (Dirichlet + also used for cloads after preprocess)
# ---------------------------------------------------------------------
boundary_schema = Schema(
    {
        # can be node-set name OR list of node IDs OR single node
        "nodes": Or(
            And(str, Use(normalize_case)),          # node set name, e.g. "NSET-1"
            And(int, Use(lambda n: [n])),           # single node
            And(list, list_of_int),                 # explicit list of nodes
        ),
        # DOF is always the 1D X DOF; keep "x" in yaml but convert to 0
        Optional("dof", default="X"): And(
            str, valid_dof_id, Use(dof_id_to_enum)
        ),
        Optional("name"): And(str, Use(normalize_case)),
        Optional("value", default=0.0): And(isnumeric, Use(float)),
        Optional("type", default=DIRICHLET): And(
            Or(str, int),
            Use(
                lambda s: (
                    {"DIRICHLET": DIRICHLET, "NEUMANN": NEUMANN}[
                        normalize_case(s)
                    ]
                    if isinstance(s, str)
                    else int(s)
                )
            ),
        ),
    }
)


# ---------------------------------------------------------------------
# Concentrated loads – same node handling, DOF -> int
# ---------------------------------------------------------------------
cload_schema = Schema(
    {
        "nodes": Or(
            And(str, Use(normalize_case)),          # node set name
            And(int, Use(lambda n: [n])),           # single node
            And(list, list_of_int),                 # list of nodes
        ),
        Optional("dof", default="X"): And(
            str, valid_dof_id, Use(dof_id_to_enum)
        ),
        Optional("name"): And(str, Use(normalize_case)),
        Optional("value", default=0.0): Use(float),
    }
)


# ---------------------------------------------------------------------
# Distributed loads (1D only, direction length == 1)
# ---------------------------------------------------------------------
def valid_dload_type(arg: str):
    return normalize_case(arg) in {"BX", "GRAV"}


dload_schema = Schema(
    {
        "elements": Or(
            And(str, Use(normalize_case)),          # element set name
            And(int, Use(lambda e: [e])),           # single element
            And(list, list_of_int),                 # list of elements
        ),
        "type": And(str, valid_dload_type, Use(normalize_case)),
        "value": Use(float),
        "direction": And(
            list,
            lambda seq: all(isnumeric(x) for x in seq),
            lambda seq: len(seq) == 1,              # 1D body load
            Use(lambda seq: [float(seq[0])]),
        ),
        Optional("name"): And(str, Use(normalize_case)),
    }
)


# ---------------------------------------------------------------------
# Element blocks
# ---------------------------------------------------------------------
def validate_element(elem: dict[str, Any]) -> bool:
    et = normalize_case(elem["type"])
    if et == "T1D1":
        # area is optional, default = 1.0
        props = elem.get("properties", {})
        schema = Schema(
            {Optional("area", default=1.0): And(isnumeric, ispositive)}
        )
        elem["properties"] = schema.validate(props)
    else:
        raise ValueError(f"Unknown element type {elem['type']!r}")
    return True


block_schema = Schema(
    And(
        {
            "name": And(str, Use(normalize_case)),
            "material": And(str, Use(normalize_case)),
            "elements": Or(
                And(str, Use(normalize_case)),      # element set name
                And(list, list_of_int),             # explicit list
            ),
            "element": {
                "type": And(str, valid_element_type, Use(normalize_case)),
                Optional("properties", default=dict()): {str: object},
            },
        },
        lambda d: validate_element(d["element"]),
    )
)


# ---------------------------------------------------------------------
# Final input schema
# ---------------------------------------------------------------------
input_schema = Schema(
    {
        "wundy": {
            "nodes": nodes_schema,
            "elements": elements_schema,
            "boundary conditions": [boundary_schema],
            "materials": [material_schema],
            "element blocks": [block_schema],
            Optional("node sets"): [nset_schema],
            Optional("element sets"): [elset_schema],
            Optional("concentrated loads"): [cload_schema],
            Optional("distributed loads"): [dload_schema],
        }
    }
)
