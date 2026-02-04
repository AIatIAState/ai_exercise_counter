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

from progam_state import ProgramState


####################################################
# Constants
####################################################

DEFAULT_MODEL = (Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task")
DEFAULT_CAMERA_INDEX = 0
DEFAULT_MAX_POSES = 1
DEFAULT_MIN_DETECTION = 0.5
DEFAULT_MIN_PRESENCE = 0.5
DEFAULT_MIN_TRACKING = 0.5
WINDOW_NAME = "Pose 3D - Pushup Counter"

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


####################################################
# Visualization and main loop
####################################################

def draw_pose_landmarks(image: np.ndarray, landmarks, color: tuple[int, int, int]) -> None:
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
    landmark_style = solutions.drawing_utils.DrawingSpec(
        color=color,
        thickness=2,
        circle_radius=2,
    )
    connection_style = solutions.drawing_utils.DrawingSpec(
        color=color,
        thickness=2,
    )
    solutions.drawing_utils.draw_landmarks(
        image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )


def annotate_frame(
    frame: np.ndarray,
    result,
    state: ProgramState,
    pose_landmarks,
) -> np.ndarray:
    annotated = frame.copy()
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

    if result.pose_landmarks:
        for landmarks in result.pose_landmarks:
            draw_pose_landmarks(annotated, landmarks, state.skeleton_color)

    if not state.current_exercise.visibility_ok():
        warning_text = "ensure whole body is in frame"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            warning_text, font, font_scale, thickness
        )
        pad_x = 16
        pad_y = 10
        box_width = text_width + pad_x * 2
        box_height = text_height + pad_y * 2
        center_x = annotated.shape[1] // 2
        center_y = annotated.shape[0] // 2
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), -1)
        text_x = int(center_x - text_width / 2)
        text_y = int(center_y + text_height / 2)
        cv2.putText(
            annotated,
            warning_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    state.current_exercise.printExerciseDetailsToScreen(annotated, pose_landmarks)

    return annotated


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        return cv2.VideoCapture(args.video)
    return cv2.VideoCapture(DEFAULT_CAMERA_INDEX)

def exercise_selection(speech_model, state: ProgramState) -> None:
    transcript = speech_model.get_transcription().lower()
    if state.update_from_transcript(transcript):
        speech_model.clear_transcript()

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

    state = ProgramState(current_exercise="pushup") # default to jumpingjacks
    speech_model = SpeechToText(state.prompt_terms(), model_size="base")
    speech_model.start_stream()

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=DEFAULT_MAX_POSES,
        min_pose_detection_confidence=DEFAULT_MIN_DETECTION,
        min_pose_presence_confidence=DEFAULT_MIN_PRESENCE,
        min_tracking_confidence=DEFAULT_MIN_TRACKING,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            exercise_selection(speech_model, state)

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

            state.current_exercise.update(world_landmarks, pose_landmarks)

            annotated = annotate_frame(frame, result, state, pose_landmarks)
            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
