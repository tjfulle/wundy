from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .element import Element
    from .material import Material


class Block:
    def __init__(
        self,
        name: str,
        coords: NDArray[float],
        connect: NDArray[int],
        element: "Element",
        material: "Material",
    ):
        self.name = name
        self.coords = np.asarray(coords)
        self.connect = np.asarray(connect)
        self.element = element
        self.material = material
        self.num_nodes = self.coords.shape[0]
        self.num_dof = self.num_nodes * self.element.dof_per_node
        # dof_map[node, dof] -> block (global) dof
        self.dof_map = np.arange(self.num_dof, dtype=int).reshape(self.num_nodes, -1)

    @property
    def active_dofs(self) -> tuple[int]:
        nft = self.element.node_freedom_table[0]
        return tuple([i for i, x in enumerate(nft) if x])

    def assemble(
        self, u: NDArray[float], du: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float]]:
        K = np.zeros((self.num_dof, self.num_dof), dtype=float)
        F = np.zeros(self.num_dof, dtype=float)
        for nodes in self.connect:
            ue = u[nodes]
            xe = self.coords[nodes]
            ke, fe = self.element.eval(xe, ue, self.material)
            eft = self.element_freedom_table(nodes)
            K[np.ix_(eft, eft)] += ke
            F[np.ix_(eft)] += fe
        return K, F

    def element_freedom_table(self, nodes: NDArray[int]) -> list[int]:
        dof_per_node = self.element.dof_per_node
        eft = [dof_per_node * node + j for node in nodes for j in range(dof_per_node)]
        return eft
