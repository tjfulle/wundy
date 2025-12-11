import io

import numpy as np

import wundy
from wundy.schemas import DIRICHLET
from wundy.schemas import NEUMANN

def test_elem_stiff():
    material = {
        "type": "ELASTIC",
        "parameters": {"E": 1}
    }
    xe = np.array([[0.0], [1.0]])
    props = {
        "properties": {"area": 1}
    }

    ke = wundy.first.elem_stiff(material, xe, props, 2)
    A, E, L = 1, 1, 1
    expected = A*E/L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(ke, expected)
    return

def test_elem_int_force():
    material = {
        "type": "ELASTIC",
        "parameters": {"E": 1}
    }
    props = {
        "properties": {"area": 1} 
    }
    
    xe = np.array([[0.0], [1.0]])
    ue = np.array([0.0, 0.1])

    A, E, L = 1, 1, 1

    expected = A * E * ((ue[1] - ue[0])/L)

    N = wundy.first.elem_int_force(props, material, xe, ue)

    assert  np.isclose(N, expected)
    return  None

def test_elem_ext_force():
    props = {
        "properties": {"area": 1}
    }
    material = {
        "type": "ELASTIC"
    }
    dload = {
        "type": "BX",
        "value": 1.0
    }

    xe = np.array([[0.0], [1.0]])
    A, q, rho, L, g = 1, 1, 0, 1, 9.81

    expected = (q + rho*A*g)*(L/2)*np.array([0.5, 0.5])

    f_ext = wundy.first.elem_ext_force(props, material, xe, dload)
    np.allclose(f_ext, expected)
    return None


