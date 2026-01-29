#!/usr/bin/env python3
"""
Run MediaPipe Pose Landmarker (3D world landmarks) on a video or live camera.
Default model: pose_landmarker_lite.task

Controls:
  - Press 'q' or ESC to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


"""
CONFIG VALUES
"""

DEFAULT_MODEL = (
    Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"
)
DEFAULT_CAMERA_INDEX = 0
DEFAULT_DOWN_ANGLE = 90.0
DEFAULT_UP_ANGLE = 160.0
DEFAULT_MAX_POSES = 1
DEFAULT_MIN_DETECTION = 0.5
DEFAULT_MIN_PRESENCE = 0.5
DEFAULT_MIN_TRACKING = 0.5
WINDOW_NAME = "Pose 3D - Pushup Counter"

class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

class PushupCounter_FSM:
    """
    Finite State Machine (FSM) for counting pushups based on elbow angle.
    States: UP and DONW. When above or below certain angle thresholds, it transitions.
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

    def update(self, angle: float | None) -> None:
        if angle is None:
            return
        self.last_angle = angle
        if self.state == "up" and angle < self.down_angle:
            self.state = "down"
        elif self.state == "down" and angle > self.up_angle:
            self.state = "up"
            self.count += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D pose tracking + pushup counter using MediaPipe Pose Landmarker."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to a .task model file (default: lite).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. If omitted, uses the default camera.",
    )
    return parser.parse_args()


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = float(np.dot(ba, bc) / denom)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def get_point(landmarks, index: int) -> np.ndarray | None:
    if landmarks is None or len(landmarks) <= index:
        return None
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_elbow_angle(world_landmarks, pose_landmarks) -> float | None:
    # Prefer 3D world landmarks, fallback to normalized 2D landmarks if needed.
    landmarks = world_landmarks if world_landmarks else pose_landmarks
    if not landmarks:
        return None

    left = compute_angle(
        get_point(landmarks, PoseIdx.LEFT_SHOULDER),
        get_point(landmarks, PoseIdx.LEFT_ELBOW),
        get_point(landmarks, PoseIdx.LEFT_WRIST),
    )
    right = compute_angle(
        get_point(landmarks, PoseIdx.RIGHT_SHOULDER),
        get_point(landmarks, PoseIdx.RIGHT_ELBOW),
        get_point(landmarks, PoseIdx.RIGHT_WRIST),
    )

    angles = [a for a in (left, right) if a is not None]
    if not angles:
        return None
    return float(sum(angles) / len(angles))


def draw_pose_landmarks(
    image: np.ndarray, landmarks, connections=POSE_CONNECTIONS
) -> None:
    if not landmarks:
        return
    height, width = image.shape[:2]

    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        x0, y0 = int(start.x * width), int(start.y * height)
        x1, y1 = int(end.x * width), int(end.y * height)
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

    for lm in landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)


def annotate_frame(
    frame: np.ndarray, result, counter: PushupCounter_FSM, angle: float | None
) -> np.ndarray:
    annotated = frame.copy()

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            draw_pose_landmarks(annotated, pose_landmarks)

    cv2.putText(
        annotated,
        f"Pushups: {counter.count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"State: {counter.state}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if angle is not None:
        cv2.putText(
            annotated,
            f"Elbow angle: {angle:.1f}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        return cv2.VideoCapture(args.video)
    return cv2.VideoCapture(DEFAULT_CAMERA_INDEX)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model not found: {model_path}")
        return 2

    cap = open_capture(args)
    if not cap.isOpened():
        print("Failed to open video source.")
        return 2

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=DEFAULT_MAX_POSES,
        min_pose_detection_confidence=DEFAULT_MIN_DETECTION,
        min_pose_presence_confidence=DEFAULT_MIN_PRESENCE,
        min_tracking_confidence=DEFAULT_MIN_TRACKING,
    )

    counter = PushupCounter_FSM()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            world_landmarks = (
                result.pose_world_landmarks[0]
                if result.pose_world_landmarks
                else None
            )
            pose_landmarks = (
                result.pose_landmarks[0] if result.pose_landmarks else None
            )

            angle = extract_elbow_angle(world_landmarks, pose_landmarks)
            counter.update(angle)

            annotated = annotate_frame(frame, result, counter, angle)
            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
