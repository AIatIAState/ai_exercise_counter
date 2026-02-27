from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np
from numpy.core.defchararray import upper

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.60
DEFAULT_UPPERARM_ABOVE_ELBOW_RATIO = 1.0
DEFAULT_STRAIGHT_ARMS_RATIO = 1.5
EPSILON = 1e-4

class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


@dataclass(frozen=True)
class PullupTelemetry:
    state: str
    upperarm_above_elbow_ratio: float
    arms_straight_ratio: float
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


class PullupCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.count = 0
        self.state = "init"
        self.telemetry = PullupTelemetry(
            state=self.state,
            upperarm_above_elbow_ratio=0,
            visibility_ok=False,
            arms_straight_ratio=0
        )

    def update(self, pose_landmarks) -> PullupTelemetry:
        required = (
            PoseIdx.LEFT_WRIST,
            PoseIdx.RIGHT_WRIST,
            PoseIdx.LEFT_ELBOW,
            PoseIdx.RIGHT_ELBOW,
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER
        )
        visibility_ok = (
            pose_landmarks is not None
            and _min_visibility(pose_landmarks, required) >= self.visibility_threshold
        )

        if not visibility_ok:
            self.telemetry = PullupTelemetry(
                state=self.state,
                visibility_ok=False,
                upperarm_above_elbow_ratio=0,
                arms_straight_ratio=0
            )
            return self.telemetry

        left_wrist = _get_landmark(pose_landmarks, PoseIdx.LEFT_WRIST)
        right_wrist = _get_landmark(pose_landmarks, PoseIdx.RIGHT_WRIST)
        left_elbow = _get_landmark(pose_landmarks, PoseIdx.LEFT_ELBOW)
        right_elbow = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ELBOW)
        left_shoulder = _get_landmark(pose_landmarks, PoseIdx.LEFT_SHOULDER)
        right_shoulder = _get_landmark(pose_landmarks, PoseIdx.RIGHT_SHOULDER)

        # Make sure all needed key points are visible
        if (left_wrist is None
            or right_wrist is None
            or left_elbow is None
            or right_elbow is None
            or left_shoulder is None
            or right_shoulder is None
        ):
            self.telemetry = PullupTelemetry(
                state=self.state,
                visibility_ok=False,
                upperarm_above_elbow_ratio=0,
                arms_straight_ratio=0
            )
            return self.telemetry


        # Starting State --> Arms Straight --> Upper Arm Above Elbow --> Increment Count --> BOHICA
        left_forearm = abs(left_elbow.y - left_wrist.y)
        right_forearm = abs(right_elbow.y - right_wrist.y)
        left_upper_arm = abs(left_shoulder.y - left_elbow.y)
        right_upper_arm = abs(right_shoulder.y - right_elbow.y)
        left_forearm_below_elbow = max(left_elbow.y - left_shoulder.y, 0)
        right_forearm_above_elbow = max(right_elbow.y - right_shoulder.y, 0)

        elbow_above_shoulders = left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y

        #Check for state transition (where left_upper arm y distance is almost 0)
        if left_upper_arm < EPSILON or right_upper_arm < EPSILON:
            self.telemetry = PullupTelemetry(
                state = self.state,
                visibility_ok=True,
                upperarm_above_elbow_ratio=self.telemetry.upperarm_above_elbow_ratio,
                arms_straight_ratio=self.telemetry.arms_straight_ratio
            )
            return self.telemetry


        if left_forearm_below_elbow < 0 or right_forearm_above_elbow < 0:
            arms_above_elbow_ratio = 0
        else:
            arms_above_elbow_ratio = min(left_forearm_below_elbow / left_forearm, right_forearm_above_elbow / right_forearm)
        arms_straight_ratio = min(right_forearm / right_upper_arm, left_forearm / left_upper_arm)


        if self.state == "init" and arms_straight_ratio < DEFAULT_STRAIGHT_ARMS_RATIO and elbow_above_shoulders:
            self.state = "down"
        elif self.state == "down" and arms_above_elbow_ratio > DEFAULT_UPPERARM_ABOVE_ELBOW_RATIO:
            self.state = "up"
            self.count += 1
        elif self.state == "up" and arms_straight_ratio < DEFAULT_STRAIGHT_ARMS_RATIO and elbow_above_shoulders:
            self.state = "down"

        self.telemetry = PullupTelemetry(
                state=self.state,
                visibility_ok=visibility_ok,
                upperarm_above_elbow_ratio=arms_above_elbow_ratio,
                arms_straight_ratio=arms_straight_ratio
            )
        return self.telemetry


class PullupExercise(Exercise):
    def __init__(
        self,
        name: str = "leg_raises",
        display_name: str = "Leg Raises",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = PullupCounter(
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
        self._draw_text(frame, f"Upper Arm Above Elbow Ratio: {telemetry.upperarm_above_elbow_ratio:.2f}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Arms Straight Ratio: {telemetry.arms_straight_ratio:.2f}", (right_x, y), align_right=True)
        y += 24

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_WRIST,
            PoseIdx.RIGHT_WRIST,
            PoseIdx.LEFT_ELBOW,
            PoseIdx.RIGHT_ELBOW,
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER
        }
