import numpy as np
import wundy

def test_neo_small_strain():
    material = {
        "type": "NEOHOOK",
        "parameters": {"E": 1, "mu": 0.5, "lam": 0}
    }
    props = {
        "properties": {"area": 1}
    }

    xe = np.array([[0.0], [1.0]])
    ue = np.array([0.0, 0.0001])

    sigma_linear = 1 * 0.0001
    N_linear = sigma_linear * 1.0

    N_neo = wundy.first.elem_int_force(props, material, xe, ue)

    assert np.isclose(N_neo, N_linear)
    return None

def test_neo_force_monotonic():
    material = {
        "type": "NEOHOOK",
        "parameters": {"E": 1, "mu": 1, "lam": 1}
    }
    props = {
        "properties": {"area": 1}
    }

    xe = np.array([[0.0], [1.0]])
    ue1 = np.array([0.0, 0.1])
    ue2 = np.array([0.0, 0.2])

    N1 = wundy.first.elem_int_force(props, material, xe, ue1)
    N2 = wundy.first.elem_int_force(props, material, xe, ue2)

    assert N2 > N1
    return None

def test_neo_non_linear_response():
    material = {
        "type": "NEOHOOK",
        "parameters": {"E": 1, "mu": 1, "lam": 1}
    }
    props = {
        "properties": {"area": 1}
    }

    xe = np.array([[0.0], [1.0]])
    ue = np.array([0.0, 0.5])
    

    N_neo = wundy.first.elem_int_force(props, material, xe, ue)
    N_lin = 1.0 * 0.5

    assert N_neo > N_lin