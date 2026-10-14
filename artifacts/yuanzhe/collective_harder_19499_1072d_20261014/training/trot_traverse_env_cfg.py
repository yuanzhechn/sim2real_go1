"""Three-height comparison on a mandatory ramp and pyramid-stair route."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .paper_env_cfg import _ramp
from .trot_two_beat_env_cfg import TrotTwoBeatCenteredEnvCfg
from .trot_two_beat_env_cfg import TwoBeatCurriculumCfg
from .walk_contact_env_cfg import configure_play
from .traverse_terrain import HfChallengeTerrainCfg, HfTraverseTerrainCfg
from . import trot_adaptive_mdp as adaptive
from . import trot_traverse_mdp as traverse


TRAVERSE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(14.0, 4.0),
    border_width=10.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.30,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    color_scheme="height",
    sub_terrains={"mandatory_traverse": HfTraverseTerrainCfg(proportion=1.0)},
)


CHALLENGE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    size=(14.0, 4.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),
    use_cache=True,
    color_scheme="none",
    sub_terrains={"steep_challenge": HfChallengeTerrainCfg(proportion=1.0)},
)


@configclass
class TraverseCurriculumCfg(TwoBeatCurriculumCfg):
    split_diagonal = _ramp("split_diagonal", -0.25, -2.0, 24_000)
    paired_swing = _ramp("paired_swing", -0.05, -0.5, 24_000)


@configclass
class TrotTraverseEnvCfg(TrotTwoBeatCenteredEnvCfg):
    curriculum: TraverseCurriculumCfg = TraverseCurriculumCfg()
    target_height: float = 0.33

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = TRAVERSE_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.8)
        self.rewards.body_height = RewTerm(func=adaptive.body_comfort, weight=5.0, params={
            "lower": self.target_height - 0.01, "upper": self.target_height + 0.01, "width": 0.035})
        self.rewards.body_low.params["minimum"] = max(0.22, self.target_height - 0.09)
        self.rewards.level_body.weight = 4.0
        self.rewards.split_diagonal = RewTerm(func=traverse.split_diagonal_penalty, weight=-0.25)
        paired_joints = SceneEntityCfg("robot", joint_names=[
            "FL_thigh_joint", "FL_calf_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_thigh_joint", "RL_calf_joint", "RR_thigh_joint", "RR_calf_joint",
        ], preserve_order=True)
        self.rewards.paired_swing = RewTerm(
            func=traverse.paired_swing_penalty, weight=-0.05, params={"asset_cfg": paired_joints})


@configclass
class TrotTraverseH30EnvCfg(TrotTraverseEnvCfg):
    target_height: float = 0.30


@configclass
class TrotTraverseH33EnvCfg(TrotTraverseEnvCfg):
    target_height: float = 0.33


@configclass
class TrotTraverseH36EnvCfg(TrotTraverseEnvCfg):
    target_height: float = 0.36


@configclass
class TrotTraverseH30EnvCfgPlay(TrotTraverseH30EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTraverseH33EnvCfgPlay(TrotTraverseH33EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTraverseH36EnvCfgPlay(TrotTraverseH36EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
