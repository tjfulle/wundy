import numpy as np
from numpy.typing import NDArray


class Material:
    type: str = "<none>"
    subclasses: list["Material"] = []

    def __init__(self, **parameters: float) -> None: ...

    def __init_subclass__(cls):
        Material.subclasses.append(cls)

    @classmethod
    def factory(cls, type: str, parameters: dict[str, float]) -> "Material":
        for subclass in cls.subclasses:
            if subclass.type.upper() == type.upper():
                return subclass(**parameters)
        raise TypeError(f"Unknown material type {type!r}")

    def eval(self, e: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
        raise NotImplementedError


class LinearElastic(Material):
    type = "ELASTIC"

    def __init__(self, **parameters: float) -> None:
        self.parameters = parameters

    def eval(self, e: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
        D = np.array([[self.parameters["E"]]], dtype=float)
        s = D[0, 0] * e
        return D, s
