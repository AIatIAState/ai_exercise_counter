"""
Abstract base class for exercises.

Each exercise needs:
- getCount(): return the current repetition count
- update(world_landmarks, pose_landmarks): update internal state based on new pose data
- printExerciseDetailsToScreen(frame, pose_landmarks): overlay exercise-specific info on video frame
"""



from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Exercise(ABC):
    def __init__(self, name: str, display_name: str, color: tuple[int, int, int]) -> None:
        self.name = name
        self.display_name = display_name
        self.color = color

    ######################################################
    # Abstract methods to be implemented by subclasses
    ######################################################
    @abstractmethod
    def update(self, world_landmarks, pose_landmarks) -> None:
        raise NotImplementedError

    @abstractmethod
    def getCount(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def printExerciseDetailsToScreen(self, frame: np.ndarray, pose_landmarks=None) -> None:
        raise NotImplementedError

    # Helper method to draw text on frame in standardized way
    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        align_right: bool = False,
        font_scale: float = 0.7,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        x, y = origin
        if align_right:
            (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = x - text_width
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


class PlaceholderExercise(Exercise):
    ########################################################
    # This is so things dont break as we build out more exercises
    ########################################################
    def update(self, world_landmarks, pose_landmarks) -> None:
        return None

    def getCount(self) -> int:
        return 0

    def printExerciseDetailsToScreen(self, frame: np.ndarray, pose_landmarks=None) -> None:
        right_x = frame.shape[1] - 10
        y = 24
        self._draw_text(frame, self.display_name, (right_x, y), align_right=True)
        y += 24
        self._draw_text(frame, "Not implemented", (right_x, y), align_right=True)
