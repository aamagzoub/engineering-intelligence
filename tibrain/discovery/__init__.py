"""TIBRAIN Discovery sub-package.

Provides generic pattern detection for identifying recurring structures
in experience data without domain-specific knowledge.
"""

from tibrain.discovery.discovery_engine import DiscoveryEngine
from tibrain.discovery.pattern import Pattern

__all__ = ["DiscoveryEngine", "Pattern"]
