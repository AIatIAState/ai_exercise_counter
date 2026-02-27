"""
Holds the state of the current exercise program.
"""

from exercise import PlaceholderExercise
from exercise_jumpingjacks import JumpingJackExercise
from exercise_planks import PlankExercise
from exercise_leg_raises import LegRaiseExercise
from exercise_pullups import PullupExercise
from exercise_pushups import PushupExercise
from exercise_situp import SitUpExercise
from exercise_squats import SquatExercise


class ProgramState:
    EXERCISE_COLORS = {
        "pushup": (0, 128, 255),      # orange
        "situp": (255, 0, 255),      # magenta
        "plank": (0, 255, 0),         # green
        "jumpingjack": (255, 255, 0), # cyan
        "squat": (0, 0, 255),         # red
        "leg_raises": (230, 0, 255),  # pink
        "pullup": (255, 0, 0)        # blue
    }
    ALIASES_BY_EXERCISE = {
        "pushup": ["pushup", "push up", "push-up", "push ups", "push-ups", "pushups", "the best exercise ever"],
        "situp": ["crunch", "situp", "sit up", "sit-up", "sit ups", "sit-ups", "situps"],
        "plank": ["plank", "planks", "planking"],
        "jumpingjack": ["jumpingjack", "jumping jack", "jumping jacks"],
        "squat": ["squat", "squats"],
        "leg_raises": ["leg raise", "leg raises", "leg raising"],
        "pullup": ["pullup", "pullups", "pull up", "pull ups", "pull-up", "pull-ups"]
    }

    def __init__(self, current_exercise: str = "pushup") -> None:
        self._exercises = {
            "pushup": PushupExercise(
                name="pushup",
                display_name="Pushup",
                color=self.EXERCISE_COLORS["pushup"],
            ),
            "situp": SitUpExercise(
                name="situp",
                display_name="Situp",
                color=self.EXERCISE_COLORS["situp"],
            ),
            "plank": PlaceholderExercise(
                name="plank",
                display_name="Plank",
                color=self.EXERCISE_COLORS["plank"],
            ),
            "jumpingjack": JumpingJackExercise(
                name="jumpingjack",
                display_name="Jumping Jack",
                color=self.EXERCISE_COLORS["jumpingjack"],
            ),
            "squat": SquatExercise(
                name="squat",
                display_name="Squat",
                color=self.EXERCISE_COLORS["squat"],
            ),
            "leg_raises": LegRaiseExercise(
                name="leg_raises",
                display_name="Leg Raises",
                color=self.EXERCISE_COLORS["leg_raises"]
            ),
            "pullup": PullupExercise(
                name="pullup",
                display_name="Pull-ups",
                color=self.EXERCISE_COLORS["pullup"]
            )
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
