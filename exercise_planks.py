from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Set

import numpy as np

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.1
DEFAULT_STRAIGHTNESS_THRESHOLD = .3
class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class PlankTelemetry:
    state: str
    straight_back: bool
    straight_legs: bool
    shoulders_above_elbows: bool
    hips_above_elbows: bool
    visibility_ok: bool
    straight_back_ratio: float
    straight_legs_ratio: float


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
        print("VALUE ERROR")
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


class PlankCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.count = 0
        self.state = "init"
        self.telemetry = PlankTelemetry(
            state=self.state,
            straight_back=False,
            straight_legs=False,
            shoulders_above_elbows=False,
            hips_above_elbows=False,
            straight_back_ratio=0,
            straight_legs_ratio=0,
            visibility_ok=False,
        )

    def update(self, pose_landmarks) -> PlankTelemetry:
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
            self.telemetry = PlankTelemetry(
                state=self.state,
                straight_back=False,
                straight_legs=False,
                shoulders_above_elbows=False,
                hips_above_elbows=False,
                straight_back_ratio=0,
                straight_legs_ratio=0,
                visibility_ok=False,
            )
            return self.telemetry

        left_shoulder = _get_landmark(pose_landmarks, PoseIdx.LEFT_SHOULDER)
        right_shoulder = _get_landmark(pose_landmarks, PoseIdx.RIGHT_SHOULDER)
        left_elbow = _get_landmark(pose_landmarks, PoseIdx.LEFT_ELBOW)
        right_elbow = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ELBOW)
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
            or left_elbow is None
            or right_elbow is None
            or left_hip is None
            or right_hip is None
            or left_knee is None
            or right_knee is None
            or left_ankle is None
            or right_ankle is None
        ):
            self.telemetry = PlankTelemetry(
                state=self.state,
                straight_back=False,
                straight_legs=False,
                shoulders_above_elbows=False,
                hips_above_elbows=False,
                straight_back_ratio=0,
                straight_legs_ratio=0,
                visibility_ok=False,
            )
            return self.telemetry

        average_hip_y = (left_hip.y + right_hip.y) / 2
        average_hip_x = (left_hip.x + right_hip.x) / 2
        average_shoulders_y = (left_shoulder.y + right_shoulder.y) / 2
        average_shoulders_x = (left_shoulder.x + right_shoulder.x) / 2
        average_knees_y = (left_knee.y + right_knee.y) / 2
        average_knees_x = (left_knee.x + right_knee.x) / 2
        average_ankles_y = (left_ankle.y + right_ankle.y) / 2
        average_ankles_x = (left_ankle.x + right_ankle.x) / 2

        # Find the truly-straight point given the x coordinate of the middle point and point slope formula of the two edge points
        desired_left_hip_point = (left_shoulder.y - left_knee.y) / (left_shoulder.y - left_knee.y) * (left_hip.x - left_shoulder.x) + left_knee.y
        desired_right_hip_point = (right_shoulder.y - right_knee.y) / (right_shoulder.y - right_knee.y) * (right_hip.x - right_shoulder.x) + right_knee.y

        left_upper_leg_distance = math.sqrt(math.pow(left_hip.y - left_knee.y, 2) + math.pow(left_hip.x - left_knee.x, 2))
        right_upper_leg_distance = math.sqrt(math.pow(right_hip.y - right_knee.y, 2) + math.pow(right_hip.x - right_knee.x, 2))

        #Find the distance ratio of the offset / total segment length
        straight_left_back_ratio = abs(left_hip.y - desired_left_hip_point) / left_upper_leg_distance
        straight_right_back_ratio = abs(right_hip.y - desired_right_hip_point) / right_upper_leg_distance
        straight_back_ratio = max(straight_left_back_ratio, straight_right_back_ratio)

        desired_left_knee_point = (left_hip.y - left_ankle.y) / (left_hip.x - left_ankle.x) * (left_knee.x - left_hip.x) + left_ankle.y
        desired_right_knee_point = (right_hip.y - right_ankle.y) / (right_hip.x - right_ankle.x) * (right_knee.x - right_hip.x) + right_ankle.y
        
        left_lower_leg_distance = math.sqrt(math.pow(left_knee.y - left_ankle.y, 2) + math.pow(left_knee.x - left_ankle.x, 2))
        right_lower_leg_distance = math.sqrt(math.pow(right_knee.y - right_ankle.y, 2) + math.pow(right_knee.x - right_ankle.x, 2))

        straight_left_legs_ratio = abs(left_knee.y - desired_left_knee_point) / left_lower_leg_distance
        straight_right_legs_ratio = abs(right_knee.y - desired_right_knee_point) / right_lower_leg_distance
        straight_legs_ratio = max(straight_left_legs_ratio, straight_right_legs_ratio)

        #Send em to conversion camp
        straight_back = straight_back_ratio < DEFAULT_STRAIGHTNESS_THRESHOLD
        straight_legs = straight_legs_ratio < DEFAULT_STRAIGHTNESS_THRESHOLD

        shoulders_above_elbows = left_shoulder.y <= left_elbow.y and right_shoulder.y <= right_elbow.y
        hips_above_elbows = left_hip.y <= left_elbow.y and right_hip.y <= right_elbow.y

        if self.state == "init" and straight_back and straight_legs and shoulders_above_elbows and hips_above_elbows:
            self.state = "up"
            self.start_time = time.time()

        elif self.state == "up" and straight_back and straight_legs and shoulders_above_elbows and hips_above_elbows:
            self.count = time.time() - self.start_time

        elif self.state == "up" and not (straight_back and straight_legs and shoulders_above_elbows and hips_above_elbows):
            self.state = "down"

        elif self.state == "down" and straight_back and straight_legs and shoulders_above_elbows and hips_above_elbows:
            self.count = 0
            self.start_time = time.time()
            self.state = "up"

        self.telemetry = PlankTelemetry(
            state=self.state,
            straight_back=straight_back,
            straight_legs=straight_legs,
            shoulders_above_elbows=shoulders_above_elbows,
            hips_above_elbows=hips_above_elbows,
            straight_back_ratio=straight_back_ratio,
            straight_legs_ratio=straight_legs_ratio,
            visibility_ok=visibility_ok,
        )
        return self.telemetry


class PlankExercise(Exercise):
    def __init__(
        self,
        name: str = "plank",
        display_name: str = "Plank",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = PlankCounter(
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
        self._draw_text(frame, f"Straight Back: {telemetry.straight_back_ratio:.2f}, {telemetry.straight_back}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Straight Legs: {telemetry.straight_legs_ratio:.2f}, {telemetry.straight_legs}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Shoulders Over Elbow: {telemetry.shoulders_above_elbows}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Hips Over Elbow: {telemetry.hips_above_elbows}", (right_x, y), align_right=True)

    def required_landmarks(self) -> Set[int]:
        return {
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER,
            PoseIdx.LEFT_ELBOW,
            PoseIdx.RIGHT_ELBOW,
            PoseIdx.LEFT_HIP,
            PoseIdx.RIGHT_HIP,
            PoseIdx.LEFT_KNEE,
            PoseIdx.RIGHT_KNEE,
            PoseIdx.LEFT_ANKLE,
            PoseIdx.RIGHT_ANKLE,
        }
