import pytest

torch = pytest.importorskip("torch")

from scene.octree_linear import MortonLinearOctree
from utils.morton import morton3d_from_grid, world_to_grid


def test_indices_follow_morton_sorting():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    levels = torch.arange(4)
    min_bound = positions.min(dim=0).values - 0.5
    max_bound = positions.max(dim=0).values + 0.5
    octree = MortonLinearOctree(positions, levels, (min_bound, max_bound))

    mask = torch.tensor([True, False, True, True])
    indices = octree.indices_from_mask(mask)
    assert indices.ndim == 1
    assert torch.equal(mask[indices], torch.ones(indices.shape[0], dtype=torch.bool))

    ix, iy, iz = world_to_grid(positions, (min_bound, max_bound), octree.max_coord)
    morton_codes = morton3d_from_grid(ix, iy, iz)
    expected_order = torch.argsort(morton_codes)
    assert torch.equal(octree.all_indices(), expected_order)
