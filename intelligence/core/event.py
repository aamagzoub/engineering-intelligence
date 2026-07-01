from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Event:
    """
    Base event for anything important that happens in the system.

    Events are used for logging, replay, GUI, statistics, and learning.
    """

    event_type: str
    payload: dict[str, Any]
    event_id: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", str(uuid4()))

        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc))