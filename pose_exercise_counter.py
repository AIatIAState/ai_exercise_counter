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
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from speech.speech_recognition import SpeechToText


####################################################
# Constants
####################################################

DEFAULT_MODEL = (Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task")
DEFAULT_CAMERA_INDEX = 0
DEFAULT_DOWN_ANGLE = 100.0 # these worked for mason
DEFAULT_UP_ANGLE = 140.0   # these worked for mason
DEFAULT_MAX_POSES = 1
DEFAULT_MIN_DETECTION = 0.5
DEFAULT_MIN_PRESENCE = 0.5
DEFAULT_MIN_TRACKING = 0.5
DEFAULT_MIN_VISIBILITY = 0.4
WINDOW_NAME = "Pose 3D - Pushup Counter"

####################################################
# Pose landmark indices
####################################################
class PoseIdx:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

####################################################
# Main pushup counter - Finite State Machine
####################################################
class PushupCounter_FSM:
    """
    Finite State Machine (FSM) for counting pushups based on elbow angle.
    States: UP and DONW. When above or below certain angle thresholds, it transitions.
    """
    def __init__(self, down_angle: float = DEFAULT_DOWN_ANGLE, up_angle: float = DEFAULT_UP_ANGLE,) -> None:
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
    parser = argparse.ArgumentParser(description="3D pose tracking + pushup counter using MediaPipe Pose Landmarker.")
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
    """
    Compute the angle B of triangle ABC
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
    # I hate that I had a lin alg quiz on a snow day of this exact problem and now i finally used it.
    return float(np.degrees(np.arccos(cos_angle)))

####################################################
# Helper functions for visibility checks
####################################################
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

####################################################
# Elbow angle calculation
####################################################

def extract_elbow_angle(world_landmarks, pose_landmarks) -> tuple[float | None, str | None]:
    angle_landmarks = world_landmarks if world_landmarks is not None else pose_landmarks
    if angle_landmarks is None:
        return None, None

    left_vis = _side_visibility(
        pose_landmarks,
        (PoseIdx.LEFT_SHOULDER, PoseIdx.LEFT_ELBOW, PoseIdx.LEFT_WRIST),
    ) if pose_landmarks is not None else 0.0
    right_vis = _side_visibility(
        pose_landmarks,
        (PoseIdx.RIGHT_SHOULDER, PoseIdx.RIGHT_ELBOW, PoseIdx.RIGHT_WRIST),
    ) if pose_landmarks is not None else 0.0

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

def get_point(landmarks, index: int) -> np.ndarray | None:
    if landmarks is None or len(landmarks) <= index:
        return None
    lm = landmarks[index]
    # hooray
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

####################################################
# Visualization and main loop
####################################################

def draw_pose_landmarks(image: np.ndarray, landmarks) -> None:
    """
    Draw "skeleton" on image

    Adapted from MediaPipe examples online.
    """
    if not landmarks:
        return
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ]
    )
    connection_style = solutions.drawing_utils.DrawingSpec(
        color=(255, 255, 255), thickness=2
    )
    solutions.drawing_utils.draw_landmarks(
        image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
        connection_drawing_spec=connection_style,
    )


def annotate_frame(
    frame: np.ndarray,
    result,
    counter: PushupCounter_FSM,
    angle: float | None,
    side: str | None,
    current_exercise: str
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    def draw_text(text: str, origin: tuple[int, int]) -> None:
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    draw_text("q to quit", (10, 24))
    draw_text(f"Count: {counter.count}", (10, 52))
    draw_text(f"State: {counter.state}", (10, 80))
    draw_text(f"Exercise {current_exercise}", (10, 100))
    if angle is None:
        draw_text("Elbow: --", (10, 128))
    else:
        side_label = f" ({side})" if side else ""
        draw_text(f"Elbow: {angle:.1f}°{side_label}", (10, 128))

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            draw_pose_landmarks(annotated, pose_landmarks)

        if angle is not None and side is not None:
            pose_landmarks = result.pose_landmarks[0]
            elbow_index = (
                PoseIdx.LEFT_ELBOW if side == "left" else PoseIdx.RIGHT_ELBOW
            )
            if len(pose_landmarks) > elbow_index:
                elbow = pose_landmarks[elbow_index]
                elbow_x = int(elbow.x * width)
                elbow_y = int(elbow.y * height)
                draw_text(f"{angle:.0f}°", (elbow_x + 8, elbow_y - 8))

    return annotated


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        return cv2.VideoCapture(args.video)
    return cv2.VideoCapture(DEFAULT_CAMERA_INDEX)

def exercise_selection(speech_model, exercises, current_exercise):
    transcript = speech_model.get_transcription().lower()
    for exercise in exercises:
        if ((exercise in transcript) or
        (exercise.replace('-', ' ') in transcript)):
            speech_model.clear_transcript()
            return exercise
    return current_exercise

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

    exercises = ["push-up", "sit-up", "plank", "jumping jack"]
    speech_model = SpeechToText(exercises, model_size="base")
    speech_model.start_stream()
    current_exercise = "push-up"

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

            current_exercise = exercise_selection(speech_model, exercises, current_exercise)

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

            angle, side = extract_elbow_angle(world_landmarks, pose_landmarks)
            counter.update(angle)

            annotated = annotate_frame(frame, result, counter, angle, side, current_exercise)
            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
