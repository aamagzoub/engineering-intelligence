from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Pattern:
    """A detected recurring pattern in experience data."""

    state_pattern: str
    action_pattern: str
    reward_outcome: str
    confidence: float
    observations: int
