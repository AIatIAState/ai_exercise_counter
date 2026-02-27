from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.3
DEFAULT_HEAD_ABOVE_KNEE_THRESHOLD = 1.6
DEFAULT_HEAD_BELOW_KNEE_THRESHOLD = 1.0
class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class SitUpTelemetry:
    state: str
    on_ground: bool
    head_above_knee_ratio: float
    visibility_ok: bool


def _visibility(landmarks, index: int) -> float:
    """
    Returns the visibility (confidence) score for a given landmark index.
    """
    if landmarks is None or len(landmarks) <= index:
        return 0.0
    lm = landmarks[index]
    visibility = getattr(lm, "visibility", 1.0)
    presence = getattr(lm, "presence", 1.0)
    try:
        return float(visibility) * float(presence)
    except (TypeError, ValueError):
        return float(visibility)


def _min_visibility(landmarks, indices: tuple[int, ...]) -> float:
    return min(_visibility(landmarks, idx) for idx in indices)


def _get_landmark(landmarks, index: int):
    if landmarks is None or len(landmarks) <= index:
        return None
    return landmarks[index]


def _get_xy(landmarks, index: int) -> np.ndarray | None:
    lm = _get_landmark(landmarks, index)
    if lm is None:
        return None
    return np.array([lm.x, lm.y], dtype=np.float32)


class SitUpCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.count = 0
        self.state = "init"
        self.telemetry = SitUpTelemetry(
            state=self.state,
            on_ground=False,
            head_above_knee_ratio=0,
            visibility_ok=False,
        )

    def update(self, pose_landmarks) -> SitUpTelemetry:
        required = (
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER,
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE,
        )
        visibility_ok = (
            pose_landmarks is not None
            and _min_visibility(pose_landmarks, required) >= self.visibility_threshold
        )

        if not visibility_ok:
            self.telemetry = SitUpTelemetry(
                state=self.state,
                on_ground=False,
                head_above_knee_ratio=0,
                visibility_ok=False,
            )
            return self.telemetry

        left_shoulder = _get_landmark(pose_landmarks, PoseIdx.LEFT_SHOULDER)
        right_shoulder = _get_landmark(pose_landmarks, PoseIdx.RIGHT_SHOULDER)
        left_hip = _get_landmark(pose_landmarks, PoseIdx.LEFT_HIP)
        right_hip = _get_landmark(pose_landmarks, PoseIdx.RIGHT_HIP)
        left_knee = _get_landmark(pose_landmarks, PoseIdx.LEFT_KNEE)
        right_knee = _get_landmark(pose_landmarks, PoseIdx.RIGHT_KNEE)
        left_ankle = _get_landmark(pose_landmarks, PoseIdx.LEFT_ANKLE)
        right_ankle = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ANKLE)


        # Make sure all needed key points are visible
        if (
            left_shoulder is None
            or right_shoulder is None
            or left_hip is None
            or right_hip is None
            or left_knee is None
            or right_knee is None
            or left_ankle is None
            or right_ankle is None
        ):
            self.telemetry = SitUpTelemetry(
                state=self.state,
                on_ground=False,
                head_above_knee_ratio=0,
                visibility_ok=False,
            )
            return self.telemetry


        on_ground = (left_ankle.y >= left_knee.y and left_hip.y >= left_knee.y) and (right_ankle.y >= right_knee.y and right_hip.y >= right_knee.y)
        height_of_knees = max(left_hip.y - left_knee.y, right_hip.y - left_knee.y)
        height_of_shoulders = max(left_hip.y - left_hip.y, right_hip.y - right_shoulder.y)
        head_above_knee_ratio = height_of_shoulders / height_of_knees

        if self.state == "init" and on_ground and head_above_knee_ratio < DEFAULT_HEAD_ABOVE_KNEE_THRESHOLD:
            self.state = "down"

        elif self.state == "down" and on_ground and head_above_knee_ratio > DEFAULT_HEAD_ABOVE_KNEE_THRESHOLD:
            self.state = "up"
            self.count += 1
        elif self.state == "up" and on_ground and head_above_knee_ratio < DEFAULT_HEAD_BELOW_KNEE_THRESHOLD:
            self.state = "down"

        self.telemetry = SitUpTelemetry(
            state=self.state,
            on_ground=on_ground,
            head_above_knee_ratio=head_above_knee_ratio,
            visibility_ok=visibility_ok,
        )
        return self.telemetry


class SitUpExercise(Exercise):
    def __init__(
        self,
        name: str = "situp",
        display_name: str = "Sit Up",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = SitUpCounter(
            visibility_threshold=visibility_threshold,
        )

    def update(self, world_landmarks, pose_landmarks) -> None:
        self._counter.update(pose_landmarks)

    def getCount(self) -> int:
        return self._counter.count

    def visibility_ok(self) -> bool:
        return self._counter.telemetry.visibility_ok

    def printExerciseDetailsToScreen(self, frame: np.ndarray, pose_landmarks=None) -> None:
        right_x = frame.shape[1] - 10
        y = 24

        telemetry = self._counter.telemetry
        self._draw_text(frame, self.display_name, (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Count: {self.getCount()}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"State: {telemetry.state}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"On Ground: {telemetry.on_ground}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Head Over Knee Ratio: {round(telemetry.head_above_knee_ratio, 2)}", (right_x, y), align_right=True)

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER,
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE,
        }
