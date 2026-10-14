"""Elevation-map locomotion with the existing posture and swing constraints."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass

from . import height_map_mdp
from .trot_natural_env_cfg import TrotNaturalAmplitudeEnvCfg
from .traverse_terrain import HfHeightMapTrainingTerrainCfg
from .walk_contact_env_cfg import configure_play


HEIGHT_MAP_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(14.0, 4.0),
    border_width=10.0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    color_scheme="none",
    sub_terrains={"height_map_route": HfHeightMapTrainingTerrainCfg(proportion=1.0)},
)


@configclass
class TrotHeightMapEnvCfg(TrotNaturalAmplitudeEnvCfg):
    """1072-D actor: 48 proprioception + 512 heights + 512 valid bits."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = HEIGHT_MAP_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.locomotion_height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            offset=RayCasterCfg.OffsetCfg(pos=(1.55, 0.0, 2.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.1, size=(3.1, 1.5), ordering="xy",
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
        )
        self.observations.policy.terrain_height_map = ObsTerm(
            func=height_map_mdp.local_height_map,
            params={
                "sensor_cfg": SceneEntityCfg("locomotion_height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
                "clip": 1.0,
            },
        )
        self.rewards.foot_clearance.func = height_map_mdp.adaptive_swing_clearance_reward
        feet = self.rewards.foot_clearance.params["asset_cfg"]
        self.rewards.foot_clearance.params = {
            "asset_cfg": feet,
            "sensor_cfg": SceneEntityCfg("locomotion_height_scanner"),
            "lookahead": 0.30, "half_width": 0.10,
            "margin": 0.03, "minimum_clearance": 0.04, "std": 0.04,
        }


@configclass
class TrotHeightMapEnvCfgPlay(TrotHeightMapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotHeightMapLightEnvCfg(TrotHeightMapEnvCfg):
    amplitude_gain: float = 0.5


@configclass
class TrotHeightMapStrongEnvCfg(TrotHeightMapEnvCfg):
    amplitude_gain: float = 2.0
