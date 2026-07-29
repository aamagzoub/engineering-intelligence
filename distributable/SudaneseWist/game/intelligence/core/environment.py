from abc import ABC, abstractmethod

from intelligence.core.action import Action
from intelligence.core.observation import Observation


class Environment(ABC):
    """Base class for environments."""

    @abstractmethod
    def observe(self, player_id: int) -> Observation:
        pass

    @abstractmethod
    def apply_action(self, action: Action) -> None:
        pass