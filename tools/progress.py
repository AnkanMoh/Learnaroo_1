from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Callable, Dict, Optional, Tuple

ProgressCB = Callable[[float, str], None]

@dataclass
class ProgressTracker:
    cb: Optional[ProgressCB] = None
    weights: Dict[str, float] = None
    expected_s: Dict[str, float] = None

    def __post_init__(self):
        self.weights = self.weights or {}
        self.expected_s = self.expected_s or {}
        self._t0 = time.time()
        self._stage_start = time.time()
        self._done: Dict[str, float] = {}

    def _emit(self, p: float, msg: str):
        if self.cb:
            self.cb(max(0.0, min(1.0, p)), msg)

    def start(self, stage: str, msg: Optional[str] = None):
        self._stage_start = time.time()
        self._emit(self.progress(stage, 0.0), msg or f"▶️ {stage}…")

    def done(self, stage: str, msg: Optional[str] = None):
        elapsed = time.time() - self._stage_start
        self._done[stage] = elapsed
        self._emit(self.progress(stage, 1.0), msg or f"✅ {stage} done ({int(elapsed)}s)")

    def progress(self, stage: str, frac: float) -> float:
        # sum weights of stages already fully completed + current stage partial
        total_w = sum(self.weights.values()) or 1.0
        p = 0.0
        for s, w in self.weights.items():
            if s == stage:
                p += w * frac
            elif s in self._done:
                p += w * 1.0
            else:
                p += 0.0
        return p / total_w

    def eta(self, current_stage: str, frac: float) -> Tuple[int, int]:
        # naive ETA using expected seconds per stage
        # returns (eta_seconds, elapsed_seconds)
        elapsed = int(time.time() - self._t0)
        remaining = 0.0
        for s, exp in self.expected_s.items():
            if s in self._done:
                continue
            if s == current_stage:
                remaining += max(0.0, exp * (1.0 - frac))
            else:
                remaining += exp
        return int(remaining), elapsed

    def update(self, stage: str, frac: float, msg: str):
        eta_s, elapsed_s = self.eta(stage, frac)
        self._emit(self.progress(stage, frac), f"{msg}  ⏳ ETA ~{eta_s}s | elapsed {elapsed_s}s")
