from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, conint, constr


# -----------------------------
# Input
# -----------------------------

Category = Literal["superhero", "dinosaur", "space battle", "robot lab", "fairytale"]


class InputRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3, max_length=400)  # type: ignore
    category: Category


# -----------------------------
# Character + Consistency
# -----------------------------

class Character(BaseModel):
    name: constr(strip_whitespace=True, min_length=1, max_length=24)  # type: ignore
    role: constr(strip_whitespace=True, min_length=1, max_length=40)  # type: ignore
    visual_style: constr(strip_whitespace=True, min_length=1, max_length=80)  # type: ignore


class CharacterBible(BaseModel):
    """
    Produced by Character Consistency Agent.
    Inject 'style_lock' into every Veo prompt to keep look consistent.
    """
    style_lock: constr(strip_whitespace=True, min_length=10, max_length=280)  # type: ignore
    character_tokens: List[constr(strip_whitespace=True, min_length=5, max_length=140)] = Field(default_factory=list)  # type: ignore
    negative_tokens: List[constr(strip_whitespace=True, min_length=3, max_length=140)] = Field(default_factory=list)  # type: ignore


# -----------------------------
# Beats / Scenes / Plan
# -----------------------------

VeoDuration = Literal[5, 6, 7, 8]


class Beat(BaseModel):
    """
    One beat = one Veo clip + one audio clip.
    Keep narration short enough to fit the duration.
    """
    idx: conint(ge=1, le=30)  # type: ignore
    duration_s: VeoDuration = 6

    title: Optional[constr(strip_whitespace=True, min_length=1, max_length=60)] = None  # type: ignore
    narration: constr(strip_whitespace=True, min_length=10, max_length=700)  # type: ignore
    visual_prompt: constr(strip_whitespace=True, min_length=8, max_length=240)  # type: ignore

    # Keep but usually empty per your rule
    on_screen_text: List[constr(strip_whitespace=True, min_length=1, max_length=40)] = Field(default_factory=list)  # type: ignore


class Scene(BaseModel):
    idx: conint(ge=1, le=12)  # type: ignore
    title: constr(strip_whitespace=True, min_length=1, max_length=80)  # type: ignore

    # NEW: beats drive audio/video generation
    beats: List[Beat] = Field(default_factory=list)

    # Backward-compat fields (optional). We won’t rely on these anymore.
    narration: Optional[str] = None
    visual_prompt: Optional[str] = None
    on_screen_text: List[str] = Field(default_factory=list)


class LessonPlan(BaseModel):
    title: constr(strip_whitespace=True, min_length=1, max_length=80)  # type: ignore
    learning_objective: constr(strip_whitespace=True, min_length=8, max_length=220)  # type: ignore
    grade_band: constr(strip_whitespace=True, min_length=3, max_length=20) = "Grades 4–6"  # type: ignore

    category: Category
    characters: List[Character] = Field(default_factory=list)

    scenes: List[Scene] = Field(default_factory=list)

    # Added later by Character Consistency Agent
    character_bible: Optional[CharacterBible] = None
