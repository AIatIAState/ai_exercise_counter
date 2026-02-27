from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.60
DEFAULT_LEGS_ABOVE_HIPS_RATIO = .6
DEFAULT_STRAIGHT_LEGS_RATIO = .8
EPSILON = 1e-4

class PoseIdx:
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class LegRaiseTelemetry:
    state: str
    legs_above_hips_ratio: float
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
        return min(float(visibility), float(presence))
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


class LegRaiseCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.count = 0
        self.state = "init"
        self.telemetry = LegRaiseTelemetry(
            state=self.state,
            legs_above_hips_ratio=0,
            visibility_ok=False,
            legs_straight_ratio=0
        )

    def update(self, pose_landmarks) -> LegRaiseTelemetry:
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
            self.telemetry = LegRaiseTelemetry(
                state=self.state,
                visibility_ok=False,
                legs_above_hips_ratio=0,
                legs_straight_ratio=0
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
            self.telemetry = LegRaiseTelemetry(
                state=self.state,
                visibility_ok=False,
                legs_above_hips_ratio=0,
                legs_straight_ratio=0
            )
            return self.telemetry


        # Starting State --> Straight Legs --> Knees above hips --> Increment Count --> BOHICA
        left_lower_leg = abs(left_ankle.y - left_knee.y)
        right_lower_leg = abs(right_ankle.y - right_knee.y)
        left_upper_leg = abs(left_knee.y - left_hip.y)
        right_upper_leg = abs(right_knee.y - right_hip.y)
        left_lower_leg_above_hip = max(left_hip.y - left_knee.y,0)
        right_lower_leg_above_hip = max(right_hip.y - right_knee.y,0)

        legs_above_hips_ratio = min(left_lower_leg_above_hip / left_lower_leg, right_lower_leg_above_hip / right_lower_leg)
        legs_straight_ratio = min(right_upper_leg / right_lower_leg, left_upper_leg / left_lower_leg)
        knees_below_hips = left_knee.y > left_hip.y and right_knee.y > right_hip.y

        #Check for state transition (where left_upper leg y distance is almost 0)
        if left_upper_leg < EPSILON or right_upper_leg < EPSILON:
            self.telemetry = LegRaiseTelemetry(
                state = self.state,
                visibility_ok=True,
                legs_above_hips_ratio=self.telemetry.legs_above_hips_ratio,
                legs_straight_ratio=self.telemetry.legs_straight_ratio
            )
            return self.telemetry

        if self.state == "init" and legs_straight_ratio > DEFAULT_STRAIGHT_LEGS_RATIO and knees_below_hips:
            self.state = "down"
        elif self.state == "down" and legs_above_hips_ratio > DEFAULT_LEGS_ABOVE_HIPS_RATIO:
            self.state = "up"
            self.count += 1
        elif self.state == "up" and legs_straight_ratio > DEFAULT_STRAIGHT_LEGS_RATIO and knees_below_hips:
            self.state = "down"

        self.telemetry = LegRaiseTelemetry(
                state=self.state,
                visibility_ok=visibility_ok,
                legs_above_hips_ratio=legs_above_hips_ratio,
                legs_straight_ratio=legs_straight_ratio
            )
        return self.telemetry


class LegRaiseExercise(Exercise):
    def __init__(
        self,
        name: str = "leg_raises",
        display_name: str = "Leg Raises",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = LegRaiseCounter(
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
        self._draw_text(frame, f"Legs Above Hips Ratio: {telemetry.legs_above_hips_ratio:.2f}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Legs Straight Ratio: {telemetry.legs_straight_ratio:.2f}", (right_x, y), align_right=True)
        y += 24

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE
        }
