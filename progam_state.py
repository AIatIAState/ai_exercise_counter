"""
Holds the state of the current exercise program.
"""

from exercise import PlaceholderExercise
from pushups import PushupExercise


class ProgramState:
    EXERCISE_COLORS = {
        "pushup": (0, 128, 255),      # orange
        "crunch": (255, 0, 255),      # magenta
        "plank": (0, 255, 0),         # green
        "jumpingjack": (255, 255, 0), # cyan
        "squat": (0, 0, 255),         # red
    }
    ALIASES_BY_EXERCISE = {
        "pushup": ["pushup", "push up", "push-up"],
        "crunch": ["crunch", "situp", "sit up", "sit-up"],
        "plank": ["plank"],
        "jumpingjack": ["jumpingjack", "jumping jack", "jumping jacks"],
        "squat": ["squat", "squats"],
    }

    def __init__(self, current_exercise: str = "pushup") -> None:
        self._exercises = {
            "pushup": PushupExercise(
                name="pushup",
                display_name="Pushup",
                color=self.EXERCISE_COLORS["pushup"],
            ),
            "crunch": PlaceholderExercise(
                name="crunch",
                display_name="Crunch",
                color=self.EXERCISE_COLORS["crunch"],
            ),
            "plank": PlaceholderExercise(
                name="plank",
                display_name="Plank",
                color=self.EXERCISE_COLORS["plank"],
            ),
            "jumpingjack": PlaceholderExercise(
                name="jumpingjack",
                display_name="Jumping Jack",
                color=self.EXERCISE_COLORS["jumpingjack"],
            ),
            "squat": PlaceholderExercise(
                name="squat",
                display_name="Squat",
                color=self.EXERCISE_COLORS["squat"],
            ),
        }

        if current_exercise not in self._exercises:
            current_exercise = "pushup"
        self._current_exercise = current_exercise

        self._alias_lookup = {}
        self._prompt_terms = []
        for exercise_name, aliases in self.ALIASES_BY_EXERCISE.items():
            for alias in aliases:
                alias = alias.lower()
                self._alias_lookup[alias] = exercise_name
                if alias not in self._prompt_terms:
                    self._prompt_terms.append(alias)

    @property
    def current_exercise(self):
        return self._exercises[self._current_exercise]

    @property
    def display_name(self) -> str:
        return self.current_exercise.display_name

    @property
    def skeleton_color(self) -> tuple[int, int, int]:
        return self.current_exercise.color

    def prompt_terms(self) -> list[str]:
        return list(self._prompt_terms)

    def update_from_transcript(self, transcript: str) -> bool:
        for alias, exercise_name in self._alias_lookup.items():
            if alias in transcript:
                if self._current_exercise != exercise_name:
                    self._current_exercise = exercise_name
                    return True
                return False
        return False
