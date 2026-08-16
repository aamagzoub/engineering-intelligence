from __future__ import annotations

from tibrain.discovery.pattern import Pattern


class DiscoveryEngine:
    """Generic pattern detection in experience sequences."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        self.confidence_threshold = confidence_threshold
        self._pattern_counts: dict[str, int] = {}
        self._total_observations: int = 0
        self._registry: dict[str, Pattern] = {}

    def observe(self, state_pattern: str, action_pattern: str, reward_outcome: str) -> None:
        """Record an observation for pattern detection."""
        key = f"{state_pattern}|{action_pattern}|{reward_outcome}"
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        self._total_observations += 1

        # Update or create pattern in registry
        confidence = self._pattern_counts[key] / self._total_observations
        if confidence >= self.confidence_threshold:
            self._registry[key] = Pattern(
                state_pattern=state_pattern,
                action_pattern=action_pattern,
                reward_outcome=reward_outcome,
                confidence=confidence,
                observations=self._pattern_counts[key],
            )
        elif key in self._registry:
            # Confidence dropped below threshold — remove
            del self._registry[key]

    def detect_patterns(self, experiences: list[tuple[str, str, str]]) -> list[Pattern]:
        """Process a batch of experiences and return patterns above threshold."""
        for state_p, action_p, reward_o in experiences:
            self.observe(state_p, action_p, reward_o)

        return [p for p in self._registry.values()
                if p.confidence >= self.confidence_threshold]

    def to_dict(self) -> dict:
        """Serialize engine state to a dictionary."""
        return {
            "pattern_counts": self._pattern_counts,
            "total_observations": self._total_observations,
            "threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscoveryEngine":
        """Deserialize engine state from a dictionary."""
        engine = cls(confidence_threshold=data.get("threshold", 0.3))
        engine._pattern_counts = data.get("pattern_counts", {})
        engine._total_observations = data.get("total_observations", 0)
        # Rebuild registry from pattern counts
        for key, count in engine._pattern_counts.items():
            confidence = count / max(engine._total_observations, 1)
            if confidence >= engine.confidence_threshold:
                parts = key.split("|")
                if len(parts) == 3:
                    engine._registry[key] = Pattern(
                        state_pattern=parts[0],
                        action_pattern=parts[1],
                        reward_outcome=parts[2],
                        confidence=confidence,
                        observations=count,
                    )
        return engine
