"""Morton coding helpers used by the linear octree implementation."""
from __future__ import annotations

from typing import Tuple

import torch


def world_to_grid(
    positions: torch.Tensor,
    aabb: Tuple[torch.Tensor, torch.Tensor],
    max_coord: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map world coordinates into a discrete integer grid."""

    min_bound, max_bound = aabb
    scale = (max_bound - min_bound).clamp(min=1e-6)
    normalized = (positions - min_bound) / scale
    normalized = normalized.clamp(0.0, 0.999999)
    grid = torch.floor(normalized * float(max_coord)).to(torch.long)
    return grid.unbind(-1)


def _expand_bits(v: torch.Tensor) -> torch.Tensor:
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def morton3d_from_grid(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
    """Interleave 3x10-bit coordinates into a Morton code."""

    ix = ix.to(torch.long)
    iy = iy.to(torch.long)
    iz = iz.to(torch.long)
    return _expand_bits(ix) | (_expand_bits(iy) << 1) | (_expand_bits(iz) << 2)
