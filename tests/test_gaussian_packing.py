import pytest

torch = pytest.importorskip("torch")

from scene.gaussian_packet import GaussianAttributeBuffer


def test_gaussian_view_to_packed_matches_index_select():
    positions = torch.randn(6, 3)
    colors = torch.randn(6, 3)
    opacities = torch.randn(6, 1)
    scales = torch.randn(6, 4)
    rotations = torch.randn(6, 4)
    mask = torch.tensor([True, False, True, False, True, False])

    buffer = GaussianAttributeBuffer(positions, colors, opacities, scales, rotations)
    view = buffer.view(mask=mask)
    packed = view.to_packed()

    expected_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    assert torch.allclose(packed["means"], torch.index_select(positions, 0, expected_indices))
    assert torch.allclose(packed["colors"], torch.index_select(colors, 0, expected_indices))
    assert torch.allclose(packed["opacities"], torch.index_select(opacities, 0, expected_indices).squeeze(-1))
    assert torch.allclose(packed["scales"], torch.index_select(scales, 0, expected_indices))
    assert torch.allclose(packed["quats"], torch.index_select(rotations, 0, expected_indices))
