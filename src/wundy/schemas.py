from typing import Any

from schema import And
from schema import Optional
from schema import Or
from schema import Schema
from schema import Use

NEUMANN = 0
DIRICHLET = 1


element_types = {"T1D1"}
bc_types = {"DIRICHLET", "NEUMANN"}


def node_freedom_table(elem_type: str) -> tuple[int, ...]:
    if normalize_case(elem_type) == "T1D1":
        return (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    raise ValueError(f"Unknown element type {elem_type!r}")


def valid_element_type(name: str) -> bool:
    return normalize_case(name) in element_types


def isnumeric(x) -> bool:
    return isinstance(x, (int, float))


def ispositive(arg: float | int) -> bool:
    return arg > 0


def list_of_type(sequence: list, type) -> bool:
    return all(isinstance(n, type) for n in sequence)


def list_of_numeric(sequence: list) -> bool:
    return all(isinstance(x, (float, int)) for x in sequence)


def list_of_int(sequence) -> bool:
    return list_of_type(sequence, int)


def list_of_list(sequence) -> bool:
    return list_of_type(sequence, list)


def normalize_case(string: str) -> str:
    return string.upper()


def dof_id_to_enum(dof: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[normalize_case(dof)]


def valid_bc_type(arg: str) -> bool:
    return normalize_case(arg) in bc_types


def bc_type_to_enum(bc_type: str) -> int:
    return {"DIRICHLET": DIRICHLET, "NEUMANN": NEUMANN}[normalize_case(bc_type)]


def valid_dof_id(dof: str):
    # extension to 2/3D: allow dof to be xyz
    return normalize_case(dof) in {"X", "Y"}


def valid_dload_type(arg: str):
    # extension to 2/3D: allow other DLOADs
    return normalize_case(arg) in {"BX", "GRAV", "QY"}


dload_profiles = {"UNIFORM", "EQUATION"}


def valid_dload_profile(arg: str) -> bool:
    return normalize_case(arg) in dload_profiles


def validate_dload_profile_fields(d: dict[str, Any]) -> bool:
    """Cross-field checks for distributed-load profiles.

    - UNIFORM  : must have "value"
    - EQUATION : must have an "expression" string
    """
    profile = normalize_case(d.get("profile", "UNIFORM"))

    if profile == "UNIFORM":
        return "value" in d

    if profile == "EQUATION":
        expr = d.get("expression")
        return isinstance(expr, str) and expr.strip() != ""

    return False


def validate_element(elem: dict[str, Any]) -> bool:
    if normalize_case(elem["type"]) == "T1D1":
        # T1D1 is used for both axial bars and Euler–Bernoulli beams.
        # - area: cross-sectional area (axial stiffness EA)
        # - I   : second moment of area (bending stiffness EI)
        schema = Schema(
            {
                Optional("area", default=1.0): And(isnumeric, ispositive),
                Optional("I"): And(isnumeric, ispositive),
            }
        )
        v = schema.validate(elem["properties"])
        elem["properties"].update(v)
    else:
        raise ValueError(f"Unknown element type {elem['type']!r}")
    return True


def validate_material_parameters(material: dict[str, dict[str, Any]]) -> bool:
    elastic = Schema(
        {
            "E": And(isnumeric, ispositive, error="E must be > 0"),
            "nu": And(
                isnumeric,
                lambda x: -1.0 <= x < 0.5,
                error="nu must be between -1 and .5",
            ),
        }
    )

    mat_type = normalize_case(material["type"])

    # Both ELASTIC and NEO_HOOKE share the same parameter requirements (E, nu)
    if mat_type in {"ELASTIC", "NEO_HOOKE", "NEO-HOOKE"}:
        elastic.validate(material["parameters"])
    else:
        raise ValueError(f"Unknown material {material['type']!r}")

    return True


nodes_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(isinstance(inner[0], int) for inner in outer),  # node label
        lambda outer: all(isinstance(f, (int, float)) for inner in outer for f in inner[1:]),
        Use(lambda outer: [[int(inner[0]), *[float(_) for _ in inner[1:]]] for inner in outer]),
    )
)

elements_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(list_of_int(inner) for inner in outer),
    )
)

nset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "nodes": And(list, list_of_int),
    },
)

elset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "elements": And(list, list_of_int),
    },
)

boundary_schema = Schema(
    And(
        {
            "nodes": Or(
                And(str, Use(normalize_case)),  # node set name
                And(int, Use(lambda n: [n])),  # single node
                And(list, list_of_int),  # list of nodes
            ),
            Optional("dof", default=0): And(str, valid_dof_id, Use(dof_id_to_enum)),
            Optional("name"): And(str, Use(normalize_case)),
            Optional("value", default=0.0): And(isnumeric, Use(float)),
            Optional("type", default=DIRICHLET): And(str, valid_bc_type, Use(bc_type_to_enum)),
        },
    )
)

cload_schema = Schema(
    And(
        {
            "nodes": Or(
                And(str, Use(normalize_case)),  # node set name
                And(int, Use(lambda n: [n])),  # single node
                And(list, list_of_int),  # list of nodes
            ),
            Optional("dof", default=0): And(str, valid_dof_id, Use(dof_id_to_enum)),
            Optional("name"): And(str, Use(normalize_case)),
            Optional("value", default=0.0): Use(float),
        },
    )
)

dload_schema = Schema(
    And(
        {
            "elements": Or(
                And(str, Use(normalize_case)),  # element set name
                And(int, Use(lambda e: [e])),  # single element
                And(list, list_of_int),  # list of elements
            ),
            "type": And(str, valid_dload_type, Use(normalize_case)),
            # For UNIFORM loads we use "value"; GRAV also uses "value".
            Optional("value", default=0.0): Use(float),
            # Profile: UNIFORM (default) or EQUATION
            Optional("profile", default="UNIFORM"): And(str, Use(normalize_case)),
            # EQUATION data
            Optional("expression"): And(str),
            "direction": And(
                list,
                list_of_numeric,
                lambda sequence: len(sequence) == 1,  # 1D problem
                Use(lambda sequence: [float(x) for x in sequence]),
            ),
            Optional("name"): And(str, Use(normalize_case)),
        },
        lambda d: validate_dload_profile_fields(d),
    )
)

material_schema = Schema(
    And(
        {
            "type": And(str, Use(normalize_case)),
            "name": And(str, Use(normalize_case)),
            "parameters": {str: object},
            Optional("density", default=0.0): And(isnumeric, ispositive),
        },
        lambda d: validate_material_parameters(d),
    )
)

block_schema = Schema(
    And(
        {
            "name": And(str, Use(normalize_case)),
            "material": And(str, Use(normalize_case)),
            "elements": Or(
                And(str, Use(normalize_case)),
                And(list, list_of_int),
            ),
            "element": {
                "type": And(str, valid_element_type, Use(normalize_case)),
                Optional("properties", default=dict()): {str: object},
            },
        },
        lambda d: validate_element(d["element"]),
    )
)

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
