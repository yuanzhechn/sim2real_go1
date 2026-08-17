"""Isaac Lab JointPositionAction 的动作映射。"""

from __future__ import annotations

import numpy as np


class Go1JointPositionMapper:
    """将策略的归一化动作转换为 12 个关节目标角度。"""

    def __init__(self, default_joint_pos: object, action_scale: float = 0.25, hip_scale_reduction: float = 1.0):
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(12)
        self.action_scale = float(action_scale)
        self.hip_scale_reduction = float(hip_scale_reduction)
        if self.hip_scale_reduction <= 0:
            raise ValueError("hip_scale_reduction 必须大于 0")

    def to_joint_target(self, action: object) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32).reshape(12)
        delta = normalized * self.action_scale
        delta[[0, 3, 6, 9]] *= self.hip_scale_reduction
        return self.default_joint_pos + delta
