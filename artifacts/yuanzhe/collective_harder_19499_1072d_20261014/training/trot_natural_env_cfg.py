"""Flexible alternating-diagonal gait and an Isaac Lab baseline comparison."""

from isaaclab.envs import mdp as common_mdp
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .paper_env_cfg import _ramp
from .trot_traverse_env_cfg import TrotTraverseEnvCfg
from .trot_traverse_env_cfg import TraverseCurriculumCfg
from .walk_contact_env_cfg import configure_play
from . import trot_adaptive_mdp as adaptive
from . import trot_traverse_mdp as traverse


@configclass
class NaturalCurriculumCfg(TraverseCurriculumCfg):
    # Keep safety and contact-quality regularizers, but do not make the joints
    # converge back to one preferred pose.  These are approximately the scale
    # of the official Isaac Lab Go1 baseline.
    action_smoothness = _ramp("action_smoothness", -0.03, -0.30, 24_000)
    air_time_variance = _ramp("air_time_variance", -0.02, -0.20, 24_000)
    base_motion = _ramp("base_motion", -0.25, -1.00, 24_000)
    joint_torque = _ramp("joint_torque", -2.0e-5, -2.0e-4, 24_000)
    joint_acceleration = _ramp("joint_acceleration", -2.5e-6, -2.5e-5, 24_000)
    joint_velocity = _ramp("joint_velocity", -3.0e-4, -3.0e-3, 24_000)
    joint_deviation = _ramp("joint_deviation", -0.005, -0.05, 24_000)
    split_diagonal = _ramp("split_diagonal", -0.10, -1.00, 24_000)
    # Disabled: matching thigh/calf angles was the source of rigid motion.
    paired_swing = None


@configclass
class TrotNaturalEnvCfg(TrotTraverseEnvCfg):
    """Natural two-beat gait: contact timing is paired, joint trajectories are free."""

    curriculum: NaturalCurriculumCfg = NaturalCurriculumCfg()
    stride_weight: float = 5.0
    tracking_weight: float = 6.0

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.25
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # A flat, zero-slope comfort band: no preferred point inside 33--36 cm.
        self.rewards.body_height = RewTerm(
            func=adaptive.body_comfort, weight=4.0,
            params={"lower": 0.33, "upper": 0.36, "width": 0.04},
        )
        self.rewards.body_low.params["minimum"] = 0.26

        # Allow independent thigh/calf motion.  Preserve smoothness, but at a
        # light baseline-like scale so it suppresses chatter rather than stride.
        self.rewards.action_smoothness.weight = -0.03
        self.rewards.air_time_variance.weight = -0.02
        self.rewards.base_motion.weight = -0.25
        self.rewards.joint_torque.weight = -2.0e-5
        self.rewards.joint_acceleration.weight = -2.5e-6
        self.rewards.joint_velocity.weight = -3.0e-4
        self.rewards.joint_deviation.weight = -0.005
        self.rewards.split_diagonal.weight = -0.10
        self.rewards.paired_swing = None

        feet = SceneEntityCfg(
            "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self.rewards.alternating_stride = RewTerm(
            func=traverse.alternating_stride_reward, weight=self.stride_weight,
            params={"asset_cfg": feet, "min_lead": 0.04, "width": 0.03},
        )
        self.rewards.planar_tracking = RewTerm(
            func=traverse.planar_tracking_reward, weight=self.tracking_weight,
            params={"asset_cfg": SceneEntityCfg("robot"), "lateral_std": 0.12, "yaw_std": 0.20},
        )


@configclass
class TrotNaturalStraightEnvCfg(TrotNaturalEnvCfg):
    tracking_weight: float = 10.0


@configclass
class NaturalPostureCurriculumCfg(NaturalCurriculumCfg):
    thigh_posture = _ramp("thigh_posture", -0.5, -2.0, 24_000)


@configclass
class TrotNaturalPostureEnvCfg(TrotNaturalStraightEnvCfg):
    """Straight policy continuation with a soft front/rear thigh posture band."""

    curriculum: NaturalPostureCurriculumCfg = NaturalPostureCurriculumCfg()
    posture_gain: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        thighs = SceneEntityCfg(
            "robot",
            joint_names=["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
            preserve_order=True,
        )
        self.rewards.thigh_posture = RewTerm(
            func=traverse.front_rear_thigh_posture_penalty,
            weight=-0.5 * self.posture_gain,
            params={"asset_cfg": thighs, "tolerance": 0.25, "front_minimum": 0.40},
        )
        for key in ("start_weight", "end_weight"):
            self.curriculum.thigh_posture.params[key] *= self.posture_gain


@configclass
class TrotNaturalPostureLightEnvCfg(TrotNaturalPostureEnvCfg):
    posture_gain: float = 0.5


@configclass
class TrotNaturalPostureStrongEnvCfg(TrotNaturalPostureEnvCfg):
    posture_gain: float = 2.0


@configclass
class NaturalAmplitudeCurriculumCfg(NaturalPostureCurriculumCfg):
    thigh_swing = _ramp("thigh_swing", -0.5, -4.0, 24_000)


@configclass
class TrotNaturalAmplitudeEnvCfg(TrotNaturalPostureStrongEnvCfg):
    """Strong posture correction plus soft front/rear thigh-amplitude matching."""

    curriculum: NaturalAmplitudeCurriculumCfg = NaturalAmplitudeCurriculumCfg()
    amplitude_gain: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        thighs = SceneEntityCfg(
            "robot",
            joint_names=["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
            preserve_order=True,
        )
        self.rewards.thigh_swing = RewTerm(
            func=traverse.front_rear_thigh_swing_penalty,
            weight=-0.5 * self.amplitude_gain,
            params={
                "asset_cfg": thighs,
                "position_tolerance": 0.05,
                "velocity_tolerance": 0.50,
                "velocity_scale": 0.02,
            },
        )
        for key in ("start_weight", "end_weight"):
            self.curriculum.thigh_swing.params[key] *= self.amplitude_gain


@configclass
class TrotNaturalAmplitudeLightEnvCfg(TrotNaturalAmplitudeEnvCfg):
    amplitude_gain: float = 0.5


@configclass
class TrotNaturalAmplitudeStrongEnvCfg(TrotNaturalAmplitudeEnvCfg):
    amplitude_gain: float = 2.0


@configclass
class TrotNaturalFlexEnvCfg(TrotNaturalEnvCfg):
    stride_weight: float = 7.0
    tracking_weight: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_smoothness.weight = -0.015
        self.rewards.joint_deviation.weight = -0.0025
        self.curriculum.action_smoothness.params["start_weight"] = -0.015
        self.curriculum.action_smoothness.params["end_weight"] = -0.15
        self.curriculum.joint_deviation.params["start_weight"] = -0.0025
        self.curriculum.joint_deviation.params["end_weight"] = -0.025


@configclass
class TrotIsaacBaselineEnvCfg(TrotNaturalEnvCfg):
    """Official Isaac Lab Go1 reward scales on our 48-D/no-height-scan route."""

    def __post_init__(self):
        super().__post_init__()
        # Remove all explicit gait/stride shaping for a genuine baseline.
        self.commands.load_metrics = None
        self.rewards.gait = None
        self.rewards.foot_clearance = None
        self.rewards.unsupported = None
        self.rewards.non_diagonal = None
        self.rewards.split_diagonal = None
        self.rewards.alternating_stride = None
        self.rewards.linear_velocity = None
        self.rewards.angular_velocity = None
        self.rewards.action_smoothness = None
        self.rewards.air_time_variance = None
        self.rewards.base_motion = None
        self.rewards.joint_torque = None
        self.rewards.joint_acceleration = None
        self.rewards.joint_velocity = None
        self.rewards.joint_deviation = None
        self.rewards.planar_tracking = None
        for name in (
            "action_smoothness", "air_time_variance", "base_motion", "joint_torque",
            "joint_acceleration", "joint_velocity", "joint_deviation", "unsupported",
            "non_diagonal", "split_diagonal",
        ):
            setattr(self.curriculum, name, None)
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=common_mdp.track_lin_vel_xy_exp, weight=1.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=common_mdp.track_ang_vel_z_exp, weight=0.75,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.lin_vel_z_l2 = RewTerm(func=common_mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.ang_vel_xy_l2 = RewTerm(func=common_mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.dof_torques_l2 = RewTerm(func=common_mdp.joint_torques_l2, weight=-2.0e-4)
        self.rewards.dof_acc_l2 = RewTerm(func=common_mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.action_rate_l2 = RewTerm(func=common_mdp.action_rate_l2, weight=-0.01)


for _cls in (
    TrotNaturalEnvCfg,
    TrotNaturalStraightEnvCfg,
    TrotNaturalFlexEnvCfg,
    TrotNaturalPostureEnvCfg,
    TrotNaturalPostureLightEnvCfg,
    TrotNaturalPostureStrongEnvCfg,
    TrotNaturalAmplitudeEnvCfg,
    TrotNaturalAmplitudeLightEnvCfg,
    TrotNaturalAmplitudeStrongEnvCfg,
    TrotIsaacBaselineEnvCfg,
):
    @configclass
    class _Play(_cls):
        def __post_init__(self):
            super().__post_init__()
            configure_play(self)
    _Play.__name__ = _cls.__name__ + "Play"
    globals()[_Play.__name__] = _Play
