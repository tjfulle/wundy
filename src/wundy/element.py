from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .material import Material


class Element:
    type: str = "<none>"
    subclasses: list["Element"] = []

    def __init__(self, **properties: Any) -> None: ...

    def __init_subclass__(cls):
        Element.subclasses.append(cls)

    @classmethod
    def factory(cls, type: str, properties: dict[str, float]) -> "Element":
        for subclass in cls.subclasses:
            if subclass.type.upper() == type.upper():
                return subclass(**properties)
        raise TypeError(f"Unknown element type {type!r}")

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        raise NotImplementedError

    @property
    def dof_per_node(self) -> int:
        return max([sum(_) for _ in self.node_freedom_table])

    def area(self, x: NDArray[float]) -> float:
        raise NotImplementedError

    def volume(self, x: NDArray[float]) -> float:
        raise NotImplementedError

    def eval(
        self,
        xe: NDArray[float],
        ue: NDArray[float],
        material: "Material",
    ) -> tuple[NDArray[float], NDArray[float]]:
        raise NotImplementedError

    def force(self, xe: NDArray[float], q: float) -> NDArray[float]:
        """Return external force due to distributed load q"""
        raise NotImplementedError


class Element1D(Element):
    def __init__(self, **properties: Any) -> None:
        self._area: float = properties.get("area", 1.0)
        self.ngauss: int = properties.get("ngauss", 2)

    def area(self, x: NDArray[float]) -> float:
        return self._area

    def volume(self, x: NDArray[float]) -> float:
        return self.area(x) * np.abs(x[1] - x[0])


class T1D2(Element1D):
    type: str = "T1D2"

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [(1, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

    def shape(self, xi: float) -> NDArray[float]:
        return np.array([1.0 - xi, 1.0 + xi]) / 2.0

    def shapegrad(self, xi: float) -> NDArray[float]:
        return np.array([-1.0, 1.0]) / 2.0

    def eval(
        self,
        xe: NDArray[float],
        ue: NDArray[float],
        material: "Material",
    ) -> tuple[NDArray[float], NDArray[float]]:
        he = xe[1, 0] - xe[0, 0]
        if np.isclose(he, 0.0):
            raise ValueError("Zero-length element detected")
        gp, wp = gauss_info(self.ngauss)
        ke = np.zeros((2, 2), dtype=float)
        fe = np.zeros(2, dtype=float)
        for i in range(self.ngauss):
            dNdxi = self.shapegrad(gp[i])
            dxdxi = np.dot(dNdxi, xe)
            dxidx = 1.0 / dxdxi.item()
            dNdx = dNdxi * dxidx
            B = dNdx[np.newaxis, :]
            e = np.dot(B, ue)
            D, s = material.eval(e)
            ke += wp[i] * self.area(xe) * np.dot(B.T, np.dot(D, B)) * dxdxi
            fe += wp[i] * self.area(xe) * np.dot(B.T, s).ravel() * dxdxi
        return ke, fe

    def force(self, xe: NDArray[float], q: float) -> NDArray[float]:
        fe = np.zeros(2, dtype=float)
        gp, wp = gauss_info(self.ngauss)
        for i in range(self.ngauss):
            xi = gp[i]
            N = self.shape(xi)
            dNdxi = self.shapegrad(xi)
            dxdxi = np.dot(dNdxi, xe)
            fe += wp[i] * np.dot(N.T, q).ravel() * dxdxi
        return fe


class B1D2(Element1D):
    """Euler-Bernouli beam element"""

    type: str = "B1D2"

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [(0, 1, 0, 1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 0, 0, 0, 0)]


class T2D2(Element1D):
    type: str = "T2D2"

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [(1, 1, 0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)]


def gauss_info(npoint: int) -> tuple[NDArray[float], NDArray[float]]:
    if npoint == 1:
        return (np.array([0.0]), np.array([2.0]))
    elif npoint == 2:
        xi = 1.0 / np.sqrt(3.0)
        return np.array([-xi, xi]), np.ones(2, dtype=float)
    else:
        raise NotImplementedError(f"gauss_info only supports npoint = 1 and 2, not {npoint}")
