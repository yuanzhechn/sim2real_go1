"""Corrected diagonal two-beat task; keep rejected experiments reproducible."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass
from .paper_env_cfg import _ramp
from .trot_load_env_cfg import TrotLoadCurriculumCfg
from .trot_adaptive_env_cfg import TrotLowAdaptiveEnvCfg
from .walk_contact_env_cfg import configure_play
from . import trot_two_beat_mdp as beat


@configclass
class TwoBeatCurriculumCfg(TrotLoadCurriculumCfg):
    unsupported = _ramp("unsupported", -1., -4., 24_000)
    non_diagonal = _ramp("non_diagonal", -.25, -1., 24_000)


@configclass
class TrotTwoBeatEnvCfg(TrotLowAdaptiveEnvCfg):
    curriculum: TwoBeatCurriculumCfg = TwoBeatCurriculumCfg()
    two_beat_constraint_gain: float = 1.

    def __post_init__(self):
        super().__post_init__()
        self.rewards.gait = RewTerm(func=beat.TwoBeatReward, weight=6., params={
            "min_dwell": .06, "credit_seconds": .30, "fade_seconds": .10})
        self.rewards.feet_air_time = None  # Never pay all four feet for a shared flight/landing.
        self.rewards.foot_clearance.func = beat.swing_clearance_reward
        self.rewards.unsupported = RewTerm(func=beat.unsupported_penalty, weight=-1.)
        self.rewards.non_diagonal = RewTerm(func=beat.non_diagonal_penalty, weight=-.25)
        self.commands.load_metrics = beat.TwoBeatMetricsCfg()
        # One-factor comparison: scale only the two anti-degeneration costs.
        for name in ("unsupported", "non_diagonal"):
            getattr(self.rewards, name).weight *= self.two_beat_constraint_gain
            term = getattr(self.curriculum, name)
            for key in ("start_weight", "end_weight"):
                term.params[key] *= self.two_beat_constraint_gain


@configclass
class TrotTwoBeatLightEnvCfg(TrotTwoBeatEnvCfg):
    two_beat_constraint_gain: float = .5


@configclass
class TrotTwoBeatStrongEnvCfg(TrotTwoBeatEnvCfg):
    two_beat_constraint_gain: float = 1.5


def configure_centered_reset_pose(cfg):
    """Put each foot in the hip's sagittal plane at reset.

    Go1's two 0.213 m leg links and -1.5 rad knee angle cancel their
    fore/aft offsets at a 0.75 rad thigh angle.  Keep the stock ab/adduction
    angles so this correction does not collapse the lateral stance width.
    This is a reset/default-action pose only, never a trajectory constraint.
    """
    cfg.scene.robot.init_state.joint_pos = {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        ".*_thigh_joint": 0.75,
        ".*_calf_joint": -1.5,
    }


@configclass
class TrotTwoBeatCenteredEnvCfg(TrotTwoBeatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_centered_reset_pose(self)


@configclass
class TrotTwoBeatCenteredLightEnvCfg(TrotTwoBeatLightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_centered_reset_pose(self)


@configclass
class TrotTwoBeatCenteredStrongEnvCfg(TrotTwoBeatStrongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_centered_reset_pose(self)


@configclass
class TrotTwoBeatEnvCfgPlay(TrotTwoBeatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTwoBeatLightEnvCfgPlay(TrotTwoBeatLightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTwoBeatStrongEnvCfgPlay(TrotTwoBeatStrongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTwoBeatCenteredEnvCfgPlay(TrotTwoBeatCenteredEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTwoBeatCenteredLightEnvCfgPlay(TrotTwoBeatCenteredLightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class TrotTwoBeatCenteredStrongEnvCfgPlay(TrotTwoBeatCenteredStrongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
