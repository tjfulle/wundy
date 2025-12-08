"""
materials.py
------------
Material routines for 1D FE solver.
"""

# ------------------------------------------------------------
# Linear Elastic
# ------------------------------------------------------------

def _get_E(material: dict) -> float:
    try:
        return float(material["parameters"]["E"])
    except KeyError:
        raise KeyError(
            f"Material dictionary must contain parameters['E'], got: {material}"
        )


def linear_elastic_tangent(material: dict, strain: float) -> float:
    return _get_E(material)


def linear_elastic_stress(material: dict, strain: float) -> float:
    E = _get_E(material)
    return E * float(strain)


# ------------------------------------------------------------
# Neo-Hookean (1D demonstration model)
# Accept both:  NEO_HOOKE   and   NEOHOOKEAN
# ------------------------------------------------------------

def neo_hookean_stress(material: dict, strain: float) -> float:
    E = _get_E(material)
    alpha = material.get("parameters", {}).get("alpha", 0.0)
    strain = float(strain)
    return E * strain * (1.0 + alpha * strain)


def neo_hookean_tangent(material: dict, strain: float) -> float:
    E = _get_E(material)
    alpha = material.get("parameters", {}).get("alpha", 0.0)
    strain = float(strain)
    return E * (1.0 + 2.0 * alpha * strain)


# ------------------------------------------------------------
# Material Dispatcher
# ------------------------------------------------------------

def _norm_type(material: dict) -> str:
    """Normalize all material type variants to a single form."""
    m = material["type"].upper().replace(" ", "").replace("-", "").replace("_", "")
    return m


def get_material_stress(material: dict, strain: float) -> float:
    mtype = _norm_type(material)

    if mtype == "ELASTIC":
        return linear_elastic_stress(material, strain)

    if mtype in {"NEOHOOKE", "NEOHOOKEAN"}:
        return neo_hookean_stress(material, strain)

    raise ValueError(f"Unknown material type: {material}")


def get_material_tangent(material: dict, strain: float) -> float:
    mtype = _norm_type(material)

    if mtype == "ELASTIC":
        return linear_elastic_tangent(material, strain)

    if mtype in {"NEOHOOKE", "NEOHOOKEAN"}:
        return neo_hookean_tangent(material, strain)

    raise ValueError(f"Unknown material type: {material}")
