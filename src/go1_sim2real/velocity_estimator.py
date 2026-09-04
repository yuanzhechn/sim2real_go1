"""基于着地足运动学约束的 Go1 机身线速度估计。"""

from __future__ import annotations

import numpy as np


class Go1ContactVelocityEstimator:
    """在足端相对地面静止假设下估计 body-frame base linear velocity。"""

    HIP_X = np.asarray([0.1881, 0.1881, -0.1881, -0.1881], dtype=np.float32)
    HIP_Y = np.asarray([0.04675, -0.04675, 0.04675, -0.04675], dtype=np.float32)
    SIDE = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    HIP_LINK = 0.08
    THIGH = 0.213
    CALF = 0.213

    def __init__(
        self,
        contact_threshold: float = 5.0,
        min_contacts: int = 2,
        filter_alpha: float = 0.2,
        max_speed: float = 3.0,
        max_no_contact_frames: int = 5,
    ) -> None:
        if contact_threshold < 0:
            raise ValueError("contact_threshold 不能为负数")
        if min_contacts < 1 or min_contacts > 4:
            raise ValueError("min_contacts 必须在 [1,4] 内")
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError("filter_alpha 必须在 (0,1] 内")
        if max_speed <= 0 or max_no_contact_frames < 0:
            raise ValueError("速度上限必须为正且失联帧数不能为负")
        self.contact_threshold = float(contact_threshold)
        self.min_contacts = int(min_contacts)
        self.filter_alpha = float(filter_alpha)
        self.max_speed = float(max_speed)
        self.max_no_contact_frames = int(max_no_contact_frames)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.contact_count = 0
        self.valid = False
        self._no_contact_frames = max_no_contact_frames + 1

    @staticmethod
    def _rx(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float32)

    @staticmethod
    def _ry(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)

    @classmethod
    def foot_position_and_jacobian(cls, leg: int, q: object) -> tuple[np.ndarray, np.ndarray]:
        q0, q1, q2 = np.asarray(q, dtype=np.float32).reshape(3)
        hip_origin = np.asarray((cls.HIP_X[leg], cls.HIP_Y[leg], 0.0), dtype=np.float32)
        rx = cls._rx(float(q0))
        thigh_origin = hip_origin + rx @ np.asarray((0.0, cls.SIDE[leg] * cls.HIP_LINK, 0.0))
        thigh_rotation = rx @ cls._ry(float(q1))
        calf_origin = thigh_origin + thigh_rotation @ np.asarray((0.0, 0.0, -cls.THIGH))
        calf_rotation = rx @ cls._ry(float(q1 + q2))
        foot = calf_origin + calf_rotation @ np.asarray((0.0, 0.0, -cls.CALF))

        hip_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        pitch_axis = rx @ np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        jacobian = np.column_stack(
            (
                np.cross(hip_axis, foot - hip_origin),
                np.cross(pitch_axis, foot - thigh_origin),
                np.cross(pitch_axis, foot - calf_origin),
            )
        ).astype(np.float32)
        return foot, jacobian

    def update(
        self,
        joint_pos: object,
        joint_vel: object,
        base_ang_vel: object,
        foot_force_est: object,
    ) -> np.ndarray:
        q = np.asarray(joint_pos, dtype=np.float32).reshape(4, 3)
        dq = np.asarray(joint_vel, dtype=np.float32).reshape(4, 3)
        omega = np.asarray(base_ang_vel, dtype=np.float32).reshape(3)
        forces = np.asarray(foot_force_est, dtype=np.float32).reshape(4)
        if not all(np.all(np.isfinite(value)) for value in (q, dq, omega, forces)):
            raise ValueError("速度估计输入包含 NaN 或 Inf")

        contacts = forces >= self.contact_threshold
        self.contact_count = int(np.sum(contacts))
        candidates = []
        for leg in np.flatnonzero(contacts):
            foot, jacobian = self.foot_position_and_jacobian(int(leg), q[leg])
            foot_velocity = jacobian @ dq[leg]
            candidates.append(-(np.cross(omega, foot) + foot_velocity))

        if self.contact_count >= self.min_contacts:
            measured = np.median(np.asarray(candidates, dtype=np.float32), axis=0)
            norm = float(np.linalg.norm(measured))
            if norm > self.max_speed:
                measured *= self.max_speed / norm
            self.velocity += self.filter_alpha * (measured - self.velocity)
            self._no_contact_frames = 0
            self.valid = True
        else:
            self._no_contact_frames += 1
            self.valid = self.valid and self._no_contact_frames <= self.max_no_contact_frames
        return self.velocity.copy()

    def reset(self) -> None:
        self.velocity.fill(0.0)
        self.contact_count = 0
        self.valid = False
        self._no_contact_frames = self.max_no_contact_frames + 1
