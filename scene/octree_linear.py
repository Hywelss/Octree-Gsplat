"""Morton sorted linear octree for efficient Gaussian queries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from utils.morton import morton3d_from_grid, world_to_grid


@dataclass
class MortonLinearOctree:
    """Linearized octree that keeps Gaussians sorted by Morton code."""

    positions: torch.Tensor
    levels: torch.Tensor
    aabb: Tuple[torch.Tensor, torch.Tensor]
    max_coord: int = 1 << 10

    def __post_init__(self) -> None:
        self.rebuild(self.positions, self.levels, self.aabb)

    def rebuild(
        self,
        positions: torch.Tensor,
        levels: torch.Tensor,
        aabb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        self.positions = positions
        self.levels = levels.view(-1)
        if aabb is not None:
            self.aabb = aabb
        if positions.numel() == 0:
            self.sorted_indices = torch.zeros((0,), dtype=torch.long, device=positions.device)
            return
        ix, iy, iz = world_to_grid(positions, self.aabb, self.max_coord)
        morton = morton3d_from_grid(ix, iy, iz)
        order = torch.argsort(morton)
        self.sorted_indices = order

    def all_indices(self) -> torch.Tensor:
        return self.sorted_indices

    def indices_from_mask(self, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.sorted_indices.numel() == 0:
            return self.sorted_indices
        if mask is None:
            return self.sorted_indices
        if mask.dtype != torch.bool:
            raise ValueError("Mask must be boolean")
        if mask.shape[0] != self.sorted_indices.shape[0]:
            raise ValueError("Mask length must match anchor count")
        mask_sorted = mask[self.sorted_indices]
        selected = torch.nonzero(mask_sorted, as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.positions.device)
        return torch.index_select(self.sorted_indices, 0, selected)

    def query_frustum(self, mask: torch.Tensor) -> torch.Tensor:
        return self.indices_from_mask(mask)

    def query_lod(self, mask: torch.Tensor) -> torch.Tensor:
        return self.indices_from_mask(mask)
