from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

from exercise import Exercise

DEFAULT_DOWN_ANGLE = 100.0
DEFAULT_UP_ANGLE = 140.0
DEFAULT_MIN_VISIBILITY = 0.3
DEFAULT_TORSO_FLAT_ABS = 0.3
DEFAULT_WRIST_BELOW_SHOULDER_ABS = 0.02


class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

class PushupCounterFSM:
    """
    Finite State Machine (FSM) for counting pushups based on elbow angle.
    States: UP and DOWN. When above or below certain angle thresholds, it transitions.
    """

    def __init__(
        self,
        down_angle: float = DEFAULT_DOWN_ANGLE,
        up_angle: float = DEFAULT_UP_ANGLE,
    ) -> None:
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.count = 0
        self.state = "up"
        self.last_angle: float | None = None

    def update(
        self,
        angle: float | None,
        can_enter_down: bool = True,
        can_count: bool = True,
    ) -> None:
        if angle is None:
            return
        self.last_angle = angle
        if self.state == "up" and angle < self.down_angle and can_enter_down:
            self.state = "down"
        elif self.state == "down" and angle > self.up_angle:
            self.state = "up"
            if can_count:
                self.count += 1


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """
    Compute the angle B of triangle ABC.
    """
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None
    cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _visibility(landmarks, index: int) -> float:
    if landmarks is None or len(landmarks) <= index:
        return 0.0
    lm = landmarks[index]
    visibility = getattr(lm, "visibility", 1.0)
    presence = getattr(lm, "presence", 1.0)
    try:
        return float(visibility) * float(presence)
    except (TypeError, ValueError):
        return float(visibility)


def _side_visibility(landmarks, indices: tuple[int, int, int]) -> float:
    return min(_visibility(landmarks, idx) for idx in indices)


def get_point(landmarks, index: int) -> np.ndarray | None:
    if landmarks is None or len(landmarks) <= index:
        return None
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _get_landmark(landmarks, index: int):
    if landmarks is None or len(landmarks) <= index:
        return None
    return landmarks[index]


def extract_elbow_angle(world_landmarks, pose_landmarks) -> tuple[float | None, str | None]:
    angle_landmarks = world_landmarks if world_landmarks is not None else pose_landmarks
    if angle_landmarks is None:
        return None, None

    left_vis = (
        _side_visibility(
            pose_landmarks,
            (PoseIdx.LEFT_SHOULDER, PoseIdx.LEFT_ELBOW, PoseIdx.LEFT_WRIST),
        )
        if pose_landmarks is not None
        else 0.0
    )
    right_vis = (
        _side_visibility(
            pose_landmarks,
            (PoseIdx.RIGHT_SHOULDER, PoseIdx.RIGHT_ELBOW, PoseIdx.RIGHT_WRIST),
        )
        if pose_landmarks is not None
        else 0.0
    )

    left_angle = compute_angle(
        get_point(angle_landmarks, PoseIdx.LEFT_SHOULDER),
        get_point(angle_landmarks, PoseIdx.LEFT_ELBOW),
        get_point(angle_landmarks, PoseIdx.LEFT_WRIST),
    )
    right_angle = compute_angle(
        get_point(angle_landmarks, PoseIdx.RIGHT_SHOULDER),
        get_point(angle_landmarks, PoseIdx.RIGHT_ELBOW),
        get_point(angle_landmarks, PoseIdx.RIGHT_WRIST),
    )

    candidates = []
    if left_angle is not None:
        candidates.append(("left", left_angle, left_vis))
    if right_angle is not None:
        candidates.append(("right", right_angle, right_vis))
    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item: (item[2] >= DEFAULT_MIN_VISIBILITY, item[2]),
        reverse=True,
    )
    side, angle, _ = candidates[0]
    return angle, side


def _side_pushup_position_flags(
    pose_landmarks,
    side: str,
) -> tuple[bool, bool, bool]:
    if side == "left":
        shoulder_idx = PoseIdx.LEFT_SHOULDER
        elbow_idx = PoseIdx.LEFT_ELBOW
        wrist_idx = PoseIdx.LEFT_WRIST
        hip_idx = PoseIdx.LEFT_HIP
    elif side == "right":
        shoulder_idx = PoseIdx.RIGHT_SHOULDER
        elbow_idx = PoseIdx.RIGHT_ELBOW
        wrist_idx = PoseIdx.RIGHT_WRIST
        hip_idx = PoseIdx.RIGHT_HIP
    else:
        return False, False, False

    shoulder = _get_landmark(pose_landmarks, shoulder_idx)
    elbow = _get_landmark(pose_landmarks, elbow_idx)
    wrist = _get_landmark(pose_landmarks, wrist_idx)
    hip = _get_landmark(pose_landmarks, hip_idx)
    if shoulder is None or elbow is None or wrist is None or hip is None:
        return False, False, False

    visibility_ok = (
        _visibility(pose_landmarks, shoulder_idx) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, elbow_idx) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, wrist_idx) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, hip_idx) >= DEFAULT_MIN_VISIBILITY
    )
    torso_flat = abs(hip.y - shoulder.y) <= DEFAULT_TORSO_FLAT_ABS
    wrists_below_shoulders = (wrist.y - shoulder.y) >= DEFAULT_WRIST_BELOW_SHOULDER_ABS

    is_in_position = visibility_ok and torso_flat and wrists_below_shoulders
    return is_in_position, torso_flat, wrists_below_shoulders


def _front_pushup_position_flags(
    pose_landmarks,
) -> tuple[bool, bool, bool]:
    left_shoulder = _get_landmark(pose_landmarks, PoseIdx.LEFT_SHOULDER)
    right_shoulder = _get_landmark(pose_landmarks, PoseIdx.RIGHT_SHOULDER)
    left_wrist = _get_landmark(pose_landmarks, PoseIdx.LEFT_WRIST)
    right_wrist = _get_landmark(pose_landmarks, PoseIdx.RIGHT_WRIST)
    left_hip = _get_landmark(pose_landmarks, PoseIdx.LEFT_HIP)
    right_hip = _get_landmark(pose_landmarks, PoseIdx.RIGHT_HIP)

    if (
        left_shoulder is None
        or right_shoulder is None
        or left_wrist is None
        or right_wrist is None
        or left_hip is None
        or right_hip is None
    ):
        return False, False, False

    visibility_ok = (
        _visibility(pose_landmarks, PoseIdx.LEFT_SHOULDER) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, PoseIdx.RIGHT_SHOULDER) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, PoseIdx.LEFT_WRIST) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, PoseIdx.RIGHT_WRIST) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, PoseIdx.LEFT_HIP) >= DEFAULT_MIN_VISIBILITY
        and _visibility(pose_landmarks, PoseIdx.RIGHT_HIP) >= DEFAULT_MIN_VISIBILITY
    )

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0
    torso_flat = abs(hip_y - shoulder_y) <= DEFAULT_TORSO_FLAT_ABS

    wrists_below_shoulders = (
        (left_wrist.y - left_shoulder.y) >= DEFAULT_WRIST_BELOW_SHOULDER_ABS
        and (right_wrist.y - right_shoulder.y) >= DEFAULT_WRIST_BELOW_SHOULDER_ABS
    )

    is_in_position = visibility_ok and torso_flat and wrists_below_shoulders
    return is_in_position, torso_flat, wrists_below_shoulders



@dataclass(frozen=True)
class PushupTelemetry:
    # any useful information for tracking pushups
    angle: float | None
    side: str | None
    elbow_index: int | None
    is_in_position: bool
    front_in_position: bool
    left_in_position: bool
    right_in_position: bool
    torso_flat: bool
    wrists_below_shoulders: bool


class PushupCounter:
    """
    A 2-state finite state machine to count pushups based on elbow angle.
    """
    def __init__(self, down_angle: float = DEFAULT_DOWN_ANGLE, up_angle: float = DEFAULT_UP_ANGLE) -> None:
        self._fsm = PushupCounterFSM(down_angle=down_angle, up_angle=up_angle)
        self.last_angle: float | None = None
        self.last_side: str | None = None
        self.last_elbow_index: int | None = None

    @property
    def count(self) -> int:
        return self._fsm.count

    @property
    def state(self) -> str:
        return self._fsm.state

    def update(self, world_landmarks, pose_landmarks) -> PushupTelemetry:
        angle, side = extract_elbow_angle(world_landmarks, pose_landmarks)
        front_in_position, front_torso_flat, front_wrists_below_shoulders = (
            _front_pushup_position_flags(pose_landmarks)
            if pose_landmarks is not None
            else (False, False, False)
        )
        left_in_position, left_torso_flat, left_wrists_below_shoulders = (
            _side_pushup_position_flags(pose_landmarks, "left")
            if pose_landmarks is not None
            else (False, False, False)
        )
        right_in_position, right_torso_flat, right_wrists_below_shoulders = (
            _side_pushup_position_flags(pose_landmarks, "right")
            if pose_landmarks is not None
            else (False, False, False)
        )
        is_in_position = front_in_position or left_in_position or right_in_position
        torso_flat = front_torso_flat or left_torso_flat or right_torso_flat
        wrists_below_shoulders = (
            front_wrists_below_shoulders
            or left_wrists_below_shoulders
            or right_wrists_below_shoulders
        )

        self._fsm.update(
            angle,
            can_enter_down=is_in_position,
            can_count=is_in_position,
        )

        elbow_index = None
        if side == "left":
            elbow_index = PoseIdx.LEFT_ELBOW
        elif side == "right":
            elbow_index = PoseIdx.RIGHT_ELBOW

        self.last_angle = angle
        self.last_side = side
        self.last_elbow_index = elbow_index

        return PushupTelemetry(
            angle=angle,
            side=side,
            elbow_index=elbow_index,
            is_in_position=is_in_position,
            front_in_position=front_in_position,
            left_in_position=left_in_position,
            right_in_position=right_in_position,
            torso_flat=torso_flat,
            wrists_below_shoulders=wrists_below_shoulders,
        )


class PushupExercise(Exercise): # PushupExercise is child of Exercise
    def __init__(
        self,
        name: str = "pushup",
        display_name: str = "Pushup",
        color: tuple[int, int, int] = (0, 128, 255),
        down_angle: float = DEFAULT_DOWN_ANGLE,
        up_angle: float = DEFAULT_UP_ANGLE,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = PushupCounter(down_angle=down_angle, up_angle=up_angle)
        self._telemetry = PushupTelemetry(
            angle=None,
            side=None,
            elbow_index=None,
            is_in_position=False,
            front_in_position=False,
            left_in_position=False,
            right_in_position=False,
            torso_flat=False,
            wrists_below_shoulders=False,
        )
        self._last_pose_landmarks = None

    def update(self, world_landmarks, pose_landmarks) -> None:
        self._last_pose_landmarks = pose_landmarks
        self._telemetry = self._counter.update(world_landmarks, pose_landmarks)

    def getCount(self) -> int:
        return self._counter.count

    def visibility_ok(self) -> bool:
        if self._last_pose_landmarks is None:
            return False
        left_ok = _side_visibility(
            self._last_pose_landmarks,
            (PoseIdx.LEFT_SHOULDER, PoseIdx.LEFT_ELBOW, PoseIdx.LEFT_WRIST),
        ) >= DEFAULT_MIN_VISIBILITY
        right_ok = _side_visibility(
            self._last_pose_landmarks,
            (PoseIdx.RIGHT_SHOULDER, PoseIdx.RIGHT_ELBOW, PoseIdx.RIGHT_WRIST),
        ) >= DEFAULT_MIN_VISIBILITY

        all_ok = True
        for idx in self.required_landmarks():
            if _visibility(self._last_pose_landmarks, idx) < DEFAULT_MIN_VISIBILITY:
                all_ok = False
                break

        return left_ok or right_ok or all_ok

    def printExerciseDetailsToScreen(self, frame: np.ndarray, pose_landmarks=None) -> None:
        height, width = frame.shape[:2]
        right_x = width - 10
        y = 24

        self._draw_text(frame, self.display_name, (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Count: {self.getCount()}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"State: {self._counter.state}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(
            frame,
            f"In position: {self._telemetry.is_in_position}",
            (right_x, y),
            align_right=True,
        )
        y += 24
        self._draw_text(frame, f"Torso flat: {self._telemetry.torso_flat}", (right_x, y), align_right=True,)
        y += 24
        self._draw_text(frame, f"Wrists below shoulders: {self._telemetry.wrists_below_shoulders}",(right_x, y), align_right=True,)
        y += 24

        if self._telemetry.angle is None:
            elbow_text = "Elbow: --"
        else:
            side_label = f" ({self._telemetry.side})" if self._telemetry.side else ""
            elbow_text = f"Elbow: {self._telemetry.angle:.1f} deg{side_label}"
        self._draw_text(frame, elbow_text, (right_x, y), align_right=True)

        if (
            pose_landmarks
            and self._telemetry.angle is not None
            and self._telemetry.elbow_index is not None
        ):
            elbow = pose_landmarks[self._telemetry.elbow_index]
            elbow_x = int(elbow.x * width)
            elbow_y = int(elbow.y * height)
            self._draw_text(frame, f"{self._telemetry.angle:.0f} deg", (elbow_x + 8, elbow_y - 8))

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER,
            PoseIdx.LEFT_ELBOW,
            PoseIdx.RIGHT_ELBOW,
            PoseIdx.LEFT_WRIST,
            PoseIdx.RIGHT_WRIST,
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
        }
