import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from gaussian_renderer import (
    _split_masked_gaussians_reference,
    _split_masked_gaussians_vectorized,
)


def _build_random_gaussian_inputs(num_anchors=5, num_offsets=4, seed=0):
    torch.manual_seed(seed)
    anchor = torch.randn(num_anchors, 3)
    grid_scaling = torch.randn(num_anchors, 6)
    color = torch.randn(num_anchors * num_offsets, 3)
    scale_rot = torch.randn(num_anchors * num_offsets, 7)
    offsets = torch.randn(num_anchors * num_offsets, 3)
    mask = torch.rand(num_anchors * num_offsets) > 0.4
    if not mask.any():
        mask[0] = True
    return anchor, grid_scaling, color, scale_rot, offsets, mask


def _compute_gaussian_outputs(split_outputs):
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = split_outputs
    if scaling_repeat.shape[0] == 0:
        zero = color.new_zeros((0, 3))
        return zero, zero, zero, zero
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = F.normalize(scale_rot[:, 3:7], dim=-1)
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets
    return xyz, color, scaling, rot


def _mock_render(xyz, color, height=4, width=4):
    num_pixels = height * width
    output = color.new_zeros((3, height, width))
    if xyz.shape[0] == 0:
        return output
    intensities = torch.sigmoid(xyz.sum(dim=-1, keepdim=True))
    contributions = color * intensities
    pixel_ids = torch.arange(contributions.shape[0], device=color.device) % num_pixels
    flat = color.new_zeros((num_pixels, 3))
    flat.index_add_(0, pixel_ids, contributions)
    return flat.t().reshape(3, height, width)


def test_vectorized_split_matches_reference_values():
    anchor, grid_scaling, color, scale_rot, offsets, mask = _build_random_gaussian_inputs(seed=42)
    reference = _split_masked_gaussians_reference(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)
    vectorized = _split_masked_gaussians_vectorized(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)

    for ref_chunk, vec_chunk in zip(reference, vectorized):
        assert torch.allclose(ref_chunk, vec_chunk)


def test_vectorized_split_handles_empty_mask():
    anchor, grid_scaling, color, scale_rot, offsets, mask = _build_random_gaussian_inputs(seed=7)
    mask[:] = False
    reference = _split_masked_gaussians_reference(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)
    vectorized = _split_masked_gaussians_vectorized(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)

    for ref_chunk, vec_chunk in zip(reference, vectorized):
        assert ref_chunk.shape == vec_chunk.shape
        assert ref_chunk.numel() == 0 and vec_chunk.numel() == 0


def test_mock_render_consistency_across_frames():
    for frame_seed in (0, 1):
        anchor, grid_scaling, color, scale_rot, offsets, mask = _build_random_gaussian_inputs(seed=frame_seed)
        reference = _split_masked_gaussians_reference(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)
        vectorized = _split_masked_gaussians_vectorized(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets=4)

        ref_xyz, ref_color, _, _ = _compute_gaussian_outputs(reference)
        vec_xyz, vec_color, _, _ = _compute_gaussian_outputs(vectorized)

        ref_frame = _mock_render(ref_xyz, ref_color)
        vec_frame = _mock_render(vec_xyz, vec_color)

        mse = torch.mean((ref_frame - vec_frame) ** 2).item()
        psnr = math.inf if mse == 0 else -10.0 * math.log10(mse + 1e-12)
        assert psnr > 80.0
