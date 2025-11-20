import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

import gaussian_renderer
from scene.gaussian_packet import GaussianAttributeBuffer


class _DummyCamera:
    def __init__(self):
        self.image_width = 8
        self.image_height = 8
        self.FoVx = 0.5
        self.FoVy = 0.5
        self.world_view_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.znear = 0.1
        self.zfar = 10.0


class _DummyGsplat:
    def __init__(self, gaussian_ids=None):
        self.kwargs = None
        self.gaussian_ids = gaussian_ids

    def rasterization(self, **kwargs):
        self.kwargs = kwargs
        means = kwargs["means"]
        if means.dim() == 2:
            num_points = means.shape[0]
            batch_shape = ()
        else:
            batch_shape = means.shape[:-2]
            num_points = means.shape[-2]
        render = torch.zeros((*batch_shape, kwargs["height"], kwargs["width"], 3))
        ids = self.gaussian_ids
        if ids is None:
            ids = torch.arange(num_points)
        means2d = torch.zeros((*batch_shape, ids.shape[0], 2))
        radii = torch.ones((*batch_shape, ids.shape[0]))
        setattr(means2d, "absgrad", torch.ones_like(means2d))
        meta = {"means2d": means2d, "radii": radii}
        if self.gaussian_ids is not None:
            meta["gaussian_ids"] = self.gaussian_ids
        return render, None, meta


def test_gsplat_rasterization_receives_packed_sparse():
    dummy_camera = _DummyCamera()
    bg_color = torch.zeros(3)
    positions = torch.randn(3, 3)
    colors = torch.rand(3, 3)
    opacities = torch.rand(3, 1)
    scales = torch.rand(3, 3)
    rotations = torch.rand(3, 4)
    extras = {
        "selection_mask": torch.ones(positions.shape[0] * 2, dtype=torch.bool),
        "neural_opacity": torch.rand(positions.shape[0] * 2, 1),
        "scaling": scales,
    }
    view = GaussianAttributeBuffer.from_tensors(positions, colors, opacities, scales, rotations, extras=extras)

    dummy_gsplat = _DummyGsplat()
    with mock.patch.object(gaussian_renderer, "_load_gsplat", return_value=dummy_gsplat):
        result = gaussian_renderer._render_with_gsplat(
            viewpoint_camera=dummy_camera,
            pc=None,
            pipe=types.SimpleNamespace(),
            bg_color=bg_color,
            scaling_modifier=1.0,
            retain_grad=False,
            attributes=view,
            is_training=True,
        )

    assert dummy_gsplat.kwargs["packed"] is True
    assert dummy_gsplat.kwargs["sparse_grad"] is True
    assert "selection_mask" in result
    assert torch.equal(result["selection_mask"], extras["selection_mask"])


def test_prefilter_handles_gaussian_id_mapping():
    dummy_camera = _DummyCamera()
    anchor_mask = torch.tensor([True, False, True])

    class _DummyPC:
        def __init__(self):
            self._anchor_mask = anchor_mask
            self.get_anchor = torch.rand(3, 3)
            self.get_scaling = torch.rand(3, 3)
            self.get_rotation = torch.rand(3, 4)
            self.get_opacity = torch.rand(3, 1)

    pc = _DummyPC()
    dummy_gsplat = _DummyGsplat(gaussian_ids=torch.tensor([0, 2]))

    with mock.patch.object(gaussian_renderer, "_load_gsplat", return_value=dummy_gsplat):
        visible = gaussian_renderer._prefilter_voxel_gsplat(
            dummy_camera, pc, pipe=types.SimpleNamespace(), scaling_modifier=1.0
        )

    assert torch.equal(visible, torch.tensor([True, False, True]))
