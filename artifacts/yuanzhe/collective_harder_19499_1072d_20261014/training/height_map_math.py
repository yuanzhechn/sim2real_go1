"""Pure tensor contract for the deployable 512-point local elevation map."""

from __future__ import annotations

import torch


HEIGHT_POINT_COUNT = 512
HEIGHT_VALUE_DIM = HEIGHT_POINT_COUNT
HEIGHT_VALIDITY_DIM = HEIGHT_POINT_COUNT
HEIGHT_MAP_OBSERVATION_DIM = HEIGHT_VALUE_DIM + HEIGHT_VALIDITY_DIM
PROPRIOCEPTION_DIM = 48
HEIGHT_MAP_POLICY_DIM = PROPRIOCEPTION_DIM + HEIGHT_MAP_OBSERVATION_DIM


def forward_clearance_score(ray_hits, feet, ground, heading, contact,
                            lookahead=0.30, half_width=0.10, margin=0.03,
                            minimum_clearance=0.04, std=0.04, foot_radius=0.025):
    """One-sided swing clearance above the highest cell in each foot's corridor.

    World-frame heading is a unit [N,2] vector. Missing corridor cells fall
    back to the foot's downward ray; invalid ground yields zero reward.
    More clearance never reduces this score. Contact/pair/credit gating is
    applied separately so standing or hopping cannot collect swing reward.
    """
    if lookahead <= 0 or half_width <= 0 or std <= 0 or min(margin, minimum_clearance, foot_radius) < 0:
        raise ValueError("Invalid forward-clearance parameters")
    if feet.shape != (ray_hits.shape[0], 4, 3) or ground.shape != feet.shape[:2] or heading.shape != (feet.shape[0], 2):
        raise ValueError("Invalid forward-clearance tensor shapes")
    delta = ray_hits[:, None, :, :2] - feet[:, :, None, :2]
    along = (delta * heading[:, None, None, :]).sum(-1)
    across = delta[..., 0] * heading[:, None, None, 1] - delta[..., 1] * heading[:, None, None, 0]
    valid = torch.isfinite(ray_hits).all(-1)[:, None, :]
    corridor = valid & (along >= -0.05) & (along <= lookahead) & (across.abs() <= half_width)
    obstacle = torch.where(corridor, ray_hits[:, None, :, 2], -torch.inf).amax(-1)
    obstacle = torch.maximum(obstacle, ground)
    required = torch.maximum(obstacle + margin, ground + minimum_clearance) + foot_radius
    finite = torch.isfinite(required) & torch.isfinite(feet).all(-1)
    deficit = torch.nan_to_num((required - feet[..., 2]).clamp_min(0), nan=1., posinf=1.)
    score = torch.exp(-(deficit / std).square()) * finite * ~contact
    return score, required


def encode_height_map(ray_hits, base_height, clip=1.0):
    """Return base-relative height values followed by one validity bit per ray.

    The first 512 values use the same sign convention as the supplied teacher:
    ``terrain_z - base_z``. Invalid depth cells are filled with zero and exposed
    explicitly by the following 512-value validity mask.
    """
    if ray_hits.ndim != 3 or ray_hits.shape[1:] != (HEIGHT_POINT_COUNT, 3):
        raise ValueError(f"Expected ray hits [N,{HEIGHT_POINT_COUNT},3]")
    if base_height.ndim == 1:
        base_height = base_height.unsqueeze(1)
    if base_height.shape != (ray_hits.shape[0], 1):
        raise ValueError("Expected base height [N] or [N,1]")
    if clip <= 0:
        raise ValueError("clip must be positive")
    valid = torch.isfinite(ray_hits).all(dim=2)
    height = (ray_hits[:, :, 2] - base_height).clamp(-clip, clip)
    height = torch.where(valid, height, torch.zeros_like(height))
    return torch.cat((height, valid.to(height.dtype)), dim=1)
