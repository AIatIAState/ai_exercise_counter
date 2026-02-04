from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exercise import Exercise

DEFAULT_VISIBILITY_THRESHOLD = 0.75
DEFAULT_HANDS_CLOSE_RATIO = 0.4
DEFAULT_HANDS_CLOSE_ABS = 0.2


class PoseIdx:
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass(frozen=True)
class JumpingJackTelemetry:
    state: str
    hands_close: bool
    elbows_above_head: bool
    hands_above_eyes: bool
    legs_straight: bool
    legs_apart: bool
    visibility_ok: bool
    hand_distance: float | None


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


class JumpingJackCounter:
    def __init__(
        self,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
        hands_close_ratio: float = DEFAULT_HANDS_CLOSE_RATIO,
        hands_close_abs: float = DEFAULT_HANDS_CLOSE_ABS,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.hands_close_ratio = hands_close_ratio
        self.hands_close_abs = hands_close_abs
        self.count = 0
        self.state = "down"
        self.telemetry = JumpingJackTelemetry(
            state=self.state,
            hands_close=False,
            elbows_above_head=False,
            hands_above_eyes=False,
            legs_straight=False,
            legs_apart=False,
            visibility_ok=False,
            hand_distance=None,
        )

    def update(self, pose_landmarks) -> JumpingJackTelemetry:
        required = (
            PoseIdx.LEFT_SHOULDER,
            PoseIdx.RIGHT_SHOULDER,
            PoseIdx.LEFT_ELBOW,
            PoseIdx.RIGHT_ELBOW,
            PoseIdx.LEFT_WRIST,
            PoseIdx.RIGHT_WRIST,
            PoseIdx.LEFT_EYE,
            PoseIdx.RIGHT_EYE,
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
            self.telemetry = JumpingJackTelemetry(
                state=self.state,
                hands_close=False,
                elbows_above_head=False,
                hands_above_eyes=False,
                legs_straight=False,
                legs_apart=False,
                visibility_ok=False,
                hand_distance=None,
            )
            return self.telemetry

        left_shoulder = _get_landmark(pose_landmarks, PoseIdx.LEFT_SHOULDER)
        right_shoulder = _get_landmark(pose_landmarks, PoseIdx.RIGHT_SHOULDER)
        left_elbow = _get_landmark(pose_landmarks, PoseIdx.LEFT_ELBOW)
        right_elbow = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ELBOW)
        left_wrist = _get_landmark(pose_landmarks, PoseIdx.LEFT_WRIST)
        right_wrist = _get_landmark(pose_landmarks, PoseIdx.RIGHT_WRIST)
        left_hip = _get_landmark(pose_landmarks, PoseIdx.LEFT_HIP)
        right_hip = _get_landmark(pose_landmarks, PoseIdx.RIGHT_HIP)
        left_knee = _get_landmark(pose_landmarks, PoseIdx.LEFT_KNEE)
        right_knee = _get_landmark(pose_landmarks, PoseIdx.RIGHT_KNEE)
        left_ankle = _get_landmark(pose_landmarks, PoseIdx.LEFT_ANKLE)
        right_ankle = _get_landmark(pose_landmarks, PoseIdx.RIGHT_ANKLE)
        left_eye = _get_landmark(pose_landmarks, PoseIdx.LEFT_EYE)
        right_eye = _get_landmark(pose_landmarks, PoseIdx.RIGHT_EYE)

        # Get xy coordinates for distance calculations
        left_wrist_xy = _get_xy(pose_landmarks, PoseIdx.LEFT_WRIST)
        right_wrist_xy = _get_xy(pose_landmarks, PoseIdx.RIGHT_WRIST)
        left_shoulder_xy = _get_xy(pose_landmarks, PoseIdx.LEFT_SHOULDER)
        right_shoulder_xy = _get_xy(pose_landmarks, PoseIdx.RIGHT_SHOULDER)
        left_hip_xy = _get_xy(pose_landmarks, PoseIdx.LEFT_HIP)
        right_hip_xy = _get_xy(pose_landmarks, PoseIdx.RIGHT_HIP)
        left_knee_xy = _get_xy(pose_landmarks, PoseIdx.LEFT_KNEE)
        right_knee_xy = _get_xy(pose_landmarks, PoseIdx.RIGHT_KNEE)
        left_ankle_xy = _get_xy(pose_landmarks, PoseIdx.LEFT_ANKLE)
        right_ankle_xy = _get_xy(pose_landmarks, PoseIdx.RIGHT_ANKLE)


        # Make sure all needed key points are visible
        if (
            left_shoulder is None
            or right_shoulder is None
            or left_elbow is None
            or right_elbow is None
            or left_wrist is None
            or right_wrist is None
            or left_hip is None
            or right_hip is None
            or left_knee is None
            or right_knee is None
            or left_ankle is None
            or right_ankle is None
            or left_eye is None
            or right_eye is None
        ):
            self.telemetry = JumpingJackTelemetry(
                state=self.state,
                hands_close=False,
                elbows_above_head=False,
                hands_above_eyes=False,
                legs_straight=False,
                legs_apart=False,
                visibility_ok=False,
                hand_distance=None,
            )
            return self.telemetry

        eye_y = min(left_eye.y, right_eye.y)

        elbows_above_head = left_elbow.y < eye_y and right_elbow.y < eye_y
        hands_above_eyes = left_wrist.y < eye_y and right_wrist.y < eye_y

        knees_above_feet = left_knee.y < left_ankle.y and right_knee.y < right_ankle.y
        hips_above_knees = left_hip.y < left_knee.y and right_hip.y < right_knee.y
        legs_straight = knees_above_feet and hips_above_knees


        hand_distance = None
        hands_close = False
        if (
            left_wrist_xy is not None
            and right_wrist_xy is not None
            and left_shoulder_xy is not None
            and right_shoulder_xy is not None
        ):
            hand_distance = float(np.linalg.norm(left_wrist_xy - right_wrist_xy))
            shoulder_width = float(np.linalg.norm(left_shoulder_xy - right_shoulder_xy))
            threshold = max(self.hands_close_abs, shoulder_width * self.hands_close_ratio)
            hands_close = hand_distance <= threshold

        #               O
        #              /|\      Hip distance should be shorter than ankle distance if legs are apart
        #           /---|---\
        #
        # hope you enjoy my ascii drawing :P
        legs_apart = False
        if (
            left_hip_xy is not None
            and right_hip_xy is not None
            and left_knee_xy is not None
            and right_knee_xy is not None
            and left_ankle_xy is not None
            and right_ankle_xy is not None
        ):
            hip_distance = float(np.linalg.norm(left_hip_xy - right_hip_xy))
            knee_distance = float(np.linalg.norm(left_knee_xy - right_knee_xy))
            ankle_distance = float(np.linalg.norm(left_ankle_xy - right_ankle_xy))
            legs_apart = hip_distance < knee_distance and hip_distance < ankle_distance

        is_up = (
            hands_above_eyes
            and elbows_above_head
            and hands_close
            and legs_straight
            and legs_apart
        )
        is_down = (
            left_wrist.y > left_shoulder.y
            and left_elbow.y > left_shoulder.y
            and right_wrist.y > right_shoulder.y
            and right_elbow.y > right_shoulder.y
            and legs_straight
        )

        if self.state == "down" and is_up:
            self.state = "up"
            self.count += 1
        elif self.state == "up" and is_down:
            self.state = "down"

        self.telemetry = JumpingJackTelemetry(
            state=self.state,
            hands_close=hands_close,
            elbows_above_head=elbows_above_head,
            hands_above_eyes=hands_above_eyes,
            legs_straight=legs_straight,
            legs_apart=legs_apart,
            visibility_ok=True,
            hand_distance=hand_distance,
        )
        return self.telemetry


class JumpingJackExercise(Exercise):
    def __init__(
        self,
        name: str = "jumpingjack",
        display_name: str = "Jumping Jack",
        color: tuple[int, int, int] = (255, 255, 0),
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
        hands_close_ratio: float = DEFAULT_HANDS_CLOSE_RATIO,
        hands_close_abs: float = DEFAULT_HANDS_CLOSE_ABS,
    ) -> None:
        super().__init__(name=name, display_name=display_name, color=color)
        self._counter = JumpingJackCounter(
            visibility_threshold=visibility_threshold,
            hands_close_ratio=hands_close_ratio,
            hands_close_abs=hands_close_abs,
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
        self._draw_text(frame, f"Legs ok: {telemetry.legs_straight}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Legs apart: {telemetry.legs_apart}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Hands above eyes: {telemetry.hands_above_eyes}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Elbows above head: {telemetry.elbows_above_head}", (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, f"Hands close: {telemetry.hands_close}", (right_x, y), align_right=True)
        y += 24
        if telemetry.hand_distance is None:
            hand_text = "Hands dist: --"
        else:
            hand_text = f"Hands dist: {telemetry.hand_distance:.3f}"
        self._draw_text(frame, hand_text, (right_x, y), align_right=True)
