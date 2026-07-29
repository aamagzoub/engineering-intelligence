from abc import ABC, abstractmethod

from intelligence.core.action import Action
from intelligence.core.observation import Observation


class Agent(ABC):
    """Base class for agents."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        pass