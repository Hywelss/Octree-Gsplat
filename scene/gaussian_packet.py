"""Utilities for representing packed Gaussian attributes for gsplat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class GaussianAttributeView:
    """Light-weight wrapper around a packed set of Gaussian attributes."""

    positions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    active_indices: Optional[torch.Tensor] = None
    extras: Optional[Dict[str, Any]] = None

    def _resolve_indices(self) -> torch.Tensor:
        if self.positions.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.positions.device)
        if self.active_indices is None:
            return torch.arange(
                self.positions.shape[0], device=self.positions.device, dtype=torch.long
            )
        return self.active_indices

    def as_tuple(self):
        indices = self._resolve_indices()
        if indices.numel() == 0 and self.positions.shape[0] == 0:
            return self.positions, self.colors, self.opacities, self.scales, self.rotations
        pos = torch.index_select(self.positions, 0, indices).contiguous()
        color = torch.index_select(self.colors, 0, indices).contiguous()
        opacity = torch.index_select(self.opacities, 0, indices).contiguous()
        scales = torch.index_select(self.scales, 0, indices).contiguous()
        rotations = torch.index_select(self.rotations, 0, indices).contiguous()
        return pos, color, opacity, scales, rotations

    def to_packed(self, scaling_modifier: float = 1.0, add_batch_dim: bool = False) -> Dict[str, torch.Tensor]:
        """
        Materialize tensors for gsplat.

        ``gsplat`` interprets any leading dimensions as batch dimensions. When
        ``sparse_grad=True`` it currently forbids batch dims, so we default to
        returning rank-2 tensors. Callers can opt-in to a leading batch dim for
        compatibility with older code paths via ``add_batch_dim=True``.
        """

        means, colors, opacities, scales, quats = self.as_tuple()

        def maybe_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if add_batch_dim else t

        packed = {
            "means": maybe_batch(means),
            "colors": maybe_batch(colors),
            "opacities": maybe_batch(opacities.squeeze(-1)),
            "scales": maybe_batch(scales * scaling_modifier),
            "quats": maybe_batch(quats),
        }
        return packed

    def get_extras(self) -> Dict[str, Any]:
        return self.extras or {}


class GaussianAttributeBuffer:
    """Container that can materialize :class:`GaussianAttributeView` objects."""

    def __init__(
        self,
        positions: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> None:
        self.positions = positions
        self.colors = colors
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations

    def view(
        self,
        indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> GaussianAttributeView:
        if indices is not None and mask is not None:
            raise ValueError("Provide either indices or mask, not both.")
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Mask must be boolean")
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.long)
        return GaussianAttributeView(
            positions=self.positions,
            colors=self.colors,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.rotations,
            active_indices=indices,
            extras=extras,
        )

    @staticmethod
    def from_tensors(
        positions: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        extras: Optional[Dict[str, Any]] = None,
    ) -> GaussianAttributeView:
        buffer = GaussianAttributeBuffer(
            positions=positions,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
        return buffer.view(extras=extras)
