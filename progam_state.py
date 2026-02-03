"""
Holds the state of the current exercise program.
"""
class ProgramState:
    EXERCISES = ("pushup", "crunch", "plank", "jumpingjack", "squat")
    DISPLAY_NAMES = {
        "pushup": "Pushup",
        "crunch": "Crunch",
        "plank": "Plank",
        "jumpingjack": "Jumping Jack",
        "squat": "Squat",
    }
    COLOR_BY_EXERCISE = {
        "pushup": (0, 128, 255),      # orange
        "crunch": (255, 0, 255),      # magenta
        "plank": (0, 255, 0),         # green
        "jumpingjack": (255, 255, 0), # cyan
        "squat": (0, 0, 255),         # red
    }
    ALIASES = {
        "pushup": "pushup",
        "push up": "pushup",
        "push-up": "pushup",
        "crunch": "crunch",
        "situp": "crunch",
        "sit up": "crunch",
        "sit-up": "crunch",
        "plank": "plank",
        "jumpingjack": "jumpingjack",
        "jumping jack": "jumpingjack",
        "jumping jacks": "jumpingjack",
        "squat": "squat",
        "squats": "squat",
    }

    def __init__(self, current_exercise: str = "pushup") -> None:
        self.current_exercise = (
            current_exercise if current_exercise in self.EXERCISES else "pushup"
        )

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAMES.get(self.current_exercise, self.current_exercise)

    @property
    def skeleton_color(self) -> tuple[int, int, int]:
        return self.COLOR_BY_EXERCISE.get(self.current_exercise, (255, 255, 255))

    def prompt_terms(self) -> list[str]:
        return [
            "pushup",
            "crunch",
            "plank",
            "jumping jack",
            "squat",
        ]

    def update_from_transcript(self, transcript: str) -> bool:
        for alias, exercise in self.ALIASES.items():
            if alias in transcript:
                if self.current_exercise != exercise:
                    self.current_exercise = exercise
                    return True
                return False
        return False