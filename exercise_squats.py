from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.75
DEFAULT_UPPER_TO_LOWER_LEG_RATIO = .8


class PoseIdx:
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class SquatTelemetry:
    state: str
    hips_above_knee: bool
    hips_bologna: bool
    legs_straight_ratio: float
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


class SquatCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.count = 0
        self.state = "init"
        self.telemetry = SquatTelemetry(
            state=self.state,
            legs_straight_ratio=0,
            visibility_ok=False,
            hips_bologna=False,
            hips_above_knee=False,
        )

    def update(self, pose_landmarks) -> SquatTelemetry:
        required = (
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE
        )
        visibility_ok = (
            pose_landmarks is not None
            and _min_visibility(pose_landmarks, required) >= self.visibility_threshold
        )

        if not visibility_ok:
            self.telemetry = SquatTelemetry(
                state=self.state,
                legs_straight_ratio=0,
                visibility_ok=False,
                hips_bologna=False,
                hips_above_knee=False,
            )
            return self.telemetry

        left_hip = _get_landmark(pose_landmarks, PoseIdx.LEFT_HIP)
        right_hip = _get_landmark(pose_landmarks, PoseIdx.RIGHT_HIP)
        left_knee = _get_landmark(pose_landmarks, PoseIdx.LEFT_KNEE)
        right_knee = _get_landmark(pose_landmarks, PoseIdx.RIGHT_KNEE)
        left_ankle = _get_landmark(pose_landmarks, PoseIdx.LEFT_ANKLE)
        right_ankle = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ANKLE)

        # Make sure all needed key points are visible
        if (left_hip is None
            or right_hip is None
            or left_knee is None
            or right_knee is None
            or left_ankle is None
            or right_ankle is None
        ):
            self.telemetry = SquatTelemetry(
                state=self.state,
                legs_straight_ratio=0,
                visibility_ok=False,
                hips_bologna=False,
                hips_above_knee=False,
            )
            return self.telemetry


        # Starting State --> Fully Standing --> Hips Below Knees --> Fully Standing --> Increment Count --> BOHICA
        knees_above_feet = left_knee.y < left_ankle.y and right_knee.y < right_ankle.y
        hips_above_knees = left_hip.y < left_knee.y and right_hip.y < right_knee.y
        hips_below_knees = left_hip.y >= left_knee.y and right_hip.y >= right_knee.y
        upper_left_leg_length = left_knee.y - left_hip.y
        upper_right_leg_length = right_knee.y - right_hip.y
        lower_left_leg_length = left_ankle.y - left_knee.y
        lower_right_leg_length = right_ankle.y - right_knee.y
        if not (knees_above_feet and hips_above_knees):
            legs_straight_ratio = 0
        else:
            legs_straight_ratio = min(upper_left_leg_length / lower_left_leg_length, upper_right_leg_length / lower_right_leg_length)

        if self.state == "init" and legs_straight_ratio > DEFAULT_UPPER_TO_LOWER_LEG_RATIO:
            self.state = "up"
        elif self.state == "down" and legs_straight_ratio > DEFAULT_UPPER_TO_LOWER_LEG_RATIO:
            self.state = "up"
            self.count += 1
        elif self.state == "up" and hips_below_knees:
            self.state = "down"

        self.telemetry = SquatTelemetry(
                state=self.state,
                legs_straight_ratio=legs_straight_ratio,
                visibility_ok=visibility_ok,
                hips_bologna=hips_below_knees,
                hips_above_knee=hips_above_knees,
            )
        return self.telemetry


class SquatExercise(Exercise):
    def __init__(
        self,
        name: str = "squat",
        display_name: str = "Squat",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = SquatCounter(
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
        self._draw_text(frame, f"Legs straight: {round(telemetry.legs_straight_ratio, 2)}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Hips Above Knee: {telemetry.hips_above_knee}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Hips Below Knee: {telemetry.hips_bologna}", (right_x, y), align_right=True)

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE
        }
