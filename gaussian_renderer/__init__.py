#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import torch.nn.functional as F
from einops import repeat

import math
from typing import Optional

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.gaussian_packet import GaussianAttributeBuffer, GaussianAttributeView
from utils.graphics_utils import fov2focal

_GSPLAT_MODULE = None


def _load_gsplat():
    global _GSPLAT_MODULE
    if _GSPLAT_MODULE is None:
        try:
            import gsplat  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Renderer backend 'gsplat' requested but the gsplat package is not installed. "
                "Install gsplat or run with '--backend diff'."
            ) from exc
        _GSPLAT_MODULE = gsplat
    return _GSPLAT_MODULE


def _get_backend(pipe) -> str:
    backend = getattr(pipe, "backend", "diff")
    backend = backend.lower()
    if backend not in {"diff", "gsplat"}:
        raise ValueError(f"Unsupported renderer backend '{backend}'. Expected 'diff' or 'gsplat'.")
    return backend


def _camera_matrices(viewpoint_camera, device, dtype):
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(device=device, dtype=dtype)
    fx = fov2focal(viewpoint_camera.FoVx, width)
    fy = fov2focal(viewpoint_camera.FoVy, height)
    K = torch.tensor(
        [
            [fx, 0.0, width * 0.5],
            [0.0, fy, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return viewmat, K


def _squeeze_leading_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove leading dimensions of size 1 while keeping the tensor connected to the autograd graph.
    """
    while tensor.dim() > 0 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def _empty_gaussian_splits(anchor: torch.Tensor, grid_scaling: torch.Tensor, color: torch.Tensor, scale_rot: torch.Tensor, offsets: torch.Tensor):
    """Create shape-consistent empty tensors for gaussian parameter splits."""
    return (
        grid_scaling.new_empty((0, grid_scaling.shape[-1])),
        anchor.new_empty((0, anchor.shape[-1])),
        color.new_empty((0, color.shape[-1])),
        scale_rot.new_empty((0, scale_rot.shape[-1])),
        offsets.new_empty((0, offsets.shape[-1])),
    )


def _split_masked_gaussians_reference(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets):
    """Reference implementation that mimics the original masking logic (used for testing)."""
    if anchor.shape[0] == 0 or num_offsets == 0 or mask.numel() == 0:
        return _empty_gaussian_splits(anchor, grid_scaling, color, scale_rot, offsets)

    mask = mask.view(-1).to(torch.bool)
    if color.shape[0] != mask.shape[0]:
        raise ValueError("Mask length must match flattened gaussian attributes")
    if not mask.any():
        return _empty_gaussian_splits(anchor, grid_scaling, color, scale_rot, offsets)

    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n c -> (n k) c', k=num_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    return masked.split([grid_scaling.shape[-1], anchor.shape[-1], color.shape[-1], scale_rot.shape[-1], offsets.shape[-1]], dim=-1)


def _split_masked_gaussians_vectorized(anchor, grid_scaling, color, scale_rot, offsets, mask, num_offsets):
    """Vectorized gather that builds gaussian tensors via advanced indexing only."""
    num_offsets = int(num_offsets)
    mask = mask.view(-1).to(torch.bool)
    total_candidates = mask.shape[0]

    if (
        num_offsets <= 0
        or anchor.shape[0] == 0
        or total_candidates == 0
        or color.shape[0] == 0
        or scale_rot.shape[0] == 0
        or offsets.shape[0] == 0
    ):
        return _empty_gaussian_splits(anchor, grid_scaling, color, scale_rot, offsets)

    if color.shape[0] != total_candidates or scale_rot.shape[0] != total_candidates or offsets.shape[0] != total_candidates:
        raise ValueError("All flattened gaussian attributes must align with the mask length")

    if not mask.any():
        return _empty_gaussian_splits(anchor, grid_scaling, color, scale_rot, offsets)

    active_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.long)

    if total_candidates % num_offsets != 0:
        raise ValueError(
            "Mask length must be divisible by num_offsets to map back to anchors"
        )
    num_anchors = total_candidates // num_offsets
    if anchor.shape[0] != num_anchors or grid_scaling.shape[0] != num_anchors:
        raise ValueError(
            "Anchor and grid scaling lengths must match mask-derived anchor count"
        )

    anchor_indices = torch.div(
        active_indices, num_offsets, rounding_mode="floor"
    )

    selected_scaling = torch.index_select(grid_scaling, 0, anchor_indices).contiguous()
    selected_anchor = torch.index_select(anchor, 0, anchor_indices).contiguous()
    selected_color = torch.index_select(color, 0, active_indices).contiguous()
    selected_scale_rot = torch.index_select(scale_rot, 0, active_indices).contiguous()
    selected_offsets = torch.index_select(offsets, 0, active_indices).contiguous()

    return selected_scaling, selected_anchor, selected_color, selected_scale_rot, selected_offsets


def generate_neural_gaussians(viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False, ape_code=-1):
    device = pc.get_anchor.device
    scaling_all = pc.get_scaling
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=device)

    active_anchor_indices = pc.gather_anchor_indices(visible_mask)
    if active_anchor_indices.numel() == 0:
        scaling_dim = scaling_all.shape[-1]
        extras = None
        if is_training:
            extras = {
                "selection_mask": torch.zeros((0,), dtype=torch.bool, device=device),
                "neural_opacity": torch.zeros((0, 1), device=device),
                "scaling": torch.zeros((0, scaling_dim), device=device),
            }
        return GaussianAttributeBuffer.from_tensors(
            positions=torch.zeros((0, 3), device=device),
            colors=torch.zeros((0, 3), device=device),
            opacities=torch.zeros((0, 1), device=device),
            scales=torch.zeros((0, scaling_dim), device=device),
            rotations=torch.zeros((0, 4), device=device),
            extras=extras,
        )

    anchor = torch.index_select(pc.get_anchor, 0, active_anchor_indices)
    feat = torch.index_select(pc.get_anchor_feat, 0, active_anchor_indices)
    level = torch.index_select(pc.get_level, 0, active_anchor_indices)
    grid_offsets = torch.index_select(pc._offset, 0, active_anchor_indices)
    grid_scaling = torch.index_select(scaling_all, 0, active_anchor_indices)

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            if isinstance(ape_code, (tuple, list)):
                ape_target = ape_code[0]
            else:
                ape_target = int(ape_code)
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_target
            appearance = pc.get_appearance(camera_indicies)
            
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = torch.index_select(pc._prog_ratio, 0, active_anchor_indices)
        transition_mask = torch.index_select(pc.transition_mask, 0, active_anchor_indices)
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    # gather gaussian parameters purely via index tensors
    (
        scaling_repeat,
        repeat_anchor,
        color,
        scale_rot,
        offsets,
    ) = _split_masked_gaussians_vectorized(
        anchor=anchor,
        grid_scaling=grid_scaling,
        color=color,
        scale_rot=scale_rot,
        offsets=offsets,
        mask=mask,
        num_offsets=pc.n_offsets,
    )
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets 

    extras = None
    if is_training:
        extras = {
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    return GaussianAttributeBuffer.from_tensors(
        positions=xyz,
        colors=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        extras=extras,
    )


def _render_with_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float,
    retain_grad: bool,
    attributes: GaussianAttributeView,
    is_training: bool,
):
    xyz, color, opacity, scaling, rot = attributes.as_tuple()
    extras = attributes.get_extras()
    selection_mask = extras.get("selection_mask") if extras else None
    neural_opacity = extras.get("neural_opacity") if extras else None
    scaling_cache = extras.get("scaling") if extras else scaling

    if xyz.numel() == 0:
        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
        rendered_image = bg_color.view(-1, 1, 1).expand(-1, height, width)
        viewspace_points = torch.zeros_like(xyz)
        visibility_filter = torch.zeros(
            (0,), dtype=torch.bool, device=xyz.device
        )
        radii = torch.zeros((0,), dtype=xyz.dtype, device=xyz.device)
        result = {
            "render": rendered_image,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }
        if is_training:
            result.update(
                {
                    "selection_mask": selection_mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling_cache,
                }
            )
        return result

    gsplat = _load_gsplat()

    device = xyz.device
    dtype = xyz.dtype
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    viewmat, K = _camera_matrices(viewpoint_camera, device, dtype)

    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()

    packed_gaussians = attributes.to_packed(scaling_modifier=scaling_modifier)
    backgrounds = bg_color.view(1, 1, -1)

    near_plane = float(getattr(viewpoint_camera, "znear", 0.01))
    far_plane = float(getattr(viewpoint_camera, "zfar", 100.0))

    render_colors, _, meta = gsplat.rasterization(
        **packed_gaussians,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        backgrounds=backgrounds,
        packed=True,
        sparse_grad=is_training,
        absgrad=retain_grad,
    )

    rendered = _squeeze_leading_dims(render_colors)
    rendered_image = rendered.permute(2, 0, 1).contiguous()

    means2d = _squeeze_leading_dims(meta["means2d"])
    viewspace_points = F.pad(means2d, (0, 1))
    if retain_grad:
        viewspace_points.retain_grad()

    raw_radii = _squeeze_leading_dims(meta["radii"])
    if raw_radii.dim() > 1:
        radii = raw_radii.amax(dim=-1)
    else:
        radii = raw_radii
    visibility_filter = radii > 0

    if retain_grad:
        absgrad = getattr(meta["means2d"], "absgrad", None)
        if absgrad is not None:
            absgrad = _squeeze_leading_dims(absgrad)
            viewspace_points._gsplat_absgrad = F.pad(absgrad, (0, 1))

    result = {
        "render": rendered_image,
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }
    if is_training:
        result.update(
            {
                "selection_mask": selection_mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling_cache,
            }
        )
    return result


def _prefilter_voxel_gsplat(viewpoint_camera, pc: GaussianModel, pipe, scaling_modifier: float):
    anchor_mask = pc._anchor_mask
    if anchor_mask is None:
        return anchor_mask

    visible_mask = anchor_mask.clone()
    if not torch.any(anchor_mask):
        return visible_mask

    # Convert the boolean mask to explicit indices so that any subsequent
    # gathers/scatters stay aligned even if the mask shape differs from the
    # flattened anchor buffer.
    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
    means3D = torch.index_select(pc.get_anchor, 0, anchor_indices)
    if means3D.numel() == 0:
        return visible_mask

    gsplat = _load_gsplat()

    device = means3D.device
    dtype = means3D.dtype
    viewmat, K = _camera_matrices(viewpoint_camera, device, dtype)
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()

    scales = torch.index_select(pc.get_scaling, 0, anchor_indices)
    if scales.dim() > 2:
        scales = scales[..., :3]
    else:
        scales = scales[:, :3]
    scales = scales * scaling_modifier

    rotations = torch.index_select(pc.get_rotation, 0, anchor_indices).unsqueeze(0)
    opacities = (
        torch.index_select(pc.get_opacity, 0, anchor_indices).view(-1).unsqueeze(0)
    )
    colors = torch.zeros((1, means3D.shape[0], 3), device=device, dtype=dtype)

    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)

    with torch.no_grad():
        _, _, meta = gsplat.rasterization(
            means=means3D.unsqueeze(0),
            quats=rotations,
            scales=scales.unsqueeze(0),
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=float(getattr(viewpoint_camera, "znear", 0.01)),
            far_plane=float(getattr(viewpoint_camera, "zfar", 100.0)),
            backgrounds=None,
            packed=True,
            sparse_grad=False,
        )

    radii = _squeeze_leading_dims(meta["radii"])
    if radii.dim() > 1:
        radii = radii.amax(dim=-1)
    visibility = radii > 0

    # Prefer gsplat-provided gaussian_ids (if present) to map the packed
    # radii/visibility back to the original anchor indices. This avoids shape
    # mismatches when gsplat internally culls or reorders the input set.
    mapped_indices = anchor_indices
    if isinstance(meta, dict):
        gaussian_ids = meta.get("gaussian_ids")
        if gaussian_ids is not None:
            gaussian_ids = _squeeze_leading_dims(gaussian_ids).to(anchor_indices.device)
            mapped_indices = torch.index_select(anchor_indices, 0, gaussian_ids.long())

    # Fall back to length alignment if gsplat returned fewer entries (e.g., after
    # aggressive culling) to keep the scatter in-bounds.
    if visibility.shape[0] != mapped_indices.shape[0]:
        min_len = min(visibility.shape[0], mapped_indices.shape[0])
        visibility = visibility[:min_len]
        mapped_indices = mapped_indices[:min_len]

    visible_mask[mapped_indices] = visibility
    return visible_mask

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training
    gaussian_view = generate_neural_gaussians(
        viewpoint_camera,
        pc,
        visible_mask,
        is_training=is_training,
        ape_code=ape_code,
    )
    extras = gaussian_view.get_extras()
    xyz, color, opacity, scaling, rot = gaussian_view.as_tuple()

    backend = _get_backend(pipe)
    if backend == "gsplat":
        return _render_with_gsplat(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            retain_grad=retain_grad,
            attributes=gaussian_view,
            is_training=is_training,
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": extras.get("selection_mask"),
            "neural_opacity": extras.get("neural_opacity"),
            "scaling": scaling,
        }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    backend = _get_backend(pipe)
    if backend == "gsplat":
        return _prefilter_voxel_gsplat(viewpoint_camera, pc, pipe, scaling_modifier)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_pure > 0
    return visible_mask
