import pytest

from wundy.materials import (
    linear_elastic_tangent,
    linear_elastic_stress,
)


def test_linear_elastic_tangent_and_stress():
    material = {
        "type": "ELASTIC",
        "name": "TEST",
        "parameters": {"E": 200.0, "nu": 0.3},
    }

    strain = 0.01  # 1% strain

    Et = linear_elastic_tangent(material, strain)
    sigma = linear_elastic_stress(material, strain)

    # dσ/dε = E
    assert Et == pytest.approx(200.0)
    # σ = E ε = 200 * 0.01 = 2
    assert sigma == pytest.approx(2.0)
