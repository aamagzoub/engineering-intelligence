"""
Centralized version management for Telecom-Native-Intelligence executables.

Follows Semantic Versioning: MAJOR.MINOR.PATCH
- MAJOR: Breaking architecture changes (e.g., TIBRAIN extraction)
- MINOR: New features (e.g., new agent, new UI panel)
- PATCH: Bug fixes, tweaks, rebuilds

The patch version auto-increments on each build via bump_patch().
"""

import json
from pathlib import Path

VERSION_FILE = Path(__file__).parent / "version.json"

# Default versions for each distributable
_DEFAULTS = {
    "WistDiscovery": {"major": 2, "minor": 0, "patch": 0},
    "SudaneseWist_PyGame": {"major": 2, "minor": 0, "patch": 0},
    "SudaneseWist": {"major": 1, "minor": 0, "patch": 0},
}


def _load() -> dict:
    """Load version data from version.json."""
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text())
    return {}


def _save(data: dict) -> None:
    """Save version data to version.json."""
    VERSION_FILE.write_text(json.dumps(data, indent=2))


def get_version(app_name: str) -> str:
    """Get the current version string for an app (e.g., '2.0.1')."""
    data = _load()
    if app_name not in data:
        data[app_name] = _DEFAULTS.get(app_name, {"major": 1, "minor": 0, "patch": 0})
        _save(data)
    v = data[app_name]
    return f"{v['major']}.{v['minor']}.{v['patch']}"


def get_version_tuple(app_name: str) -> tuple[int, int, int]:
    """Get the current version as a (major, minor, patch) tuple."""
    data = _load()
    if app_name not in data:
        data[app_name] = _DEFAULTS.get(app_name, {"major": 1, "minor": 0, "patch": 0})
        _save(data)
    v = data[app_name]
    return (v["major"], v["minor"], v["patch"])


def bump_patch(app_name: str) -> str:
    """Increment the patch version and return the new version string."""
    data = _load()
    if app_name not in data:
        data[app_name] = _DEFAULTS.get(app_name, {"major": 1, "minor": 0, "patch": 0})
    data[app_name]["patch"] += 1
    _save(data)
    v = data[app_name]
    return f"{v['major']}.{v['minor']}.{v['patch']}"


def bump_minor(app_name: str) -> str:
    """Increment minor version, reset patch to 0."""
    data = _load()
    if app_name not in data:
        data[app_name] = _DEFAULTS.get(app_name, {"major": 1, "minor": 0, "patch": 0})
    data[app_name]["minor"] += 1
    data[app_name]["patch"] = 0
    _save(data)
    v = data[app_name]
    return f"{v['major']}.{v['minor']}.{v['patch']}"


def bump_major(app_name: str) -> str:
    """Increment major version, reset minor and patch to 0."""
    data = _load()
    if app_name not in data:
        data[app_name] = _DEFAULTS.get(app_name, {"major": 1, "minor": 0, "patch": 0})
    data[app_name]["major"] += 1
    data[app_name]["minor"] = 0
    data[app_name]["patch"] = 0
    _save(data)
    v = data[app_name]
    return f"{v['major']}.{v['minor']}.{v['patch']}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python version.py <app_name> [bump_patch|bump_minor|bump_major]")
        print("\nCurrent versions:")
        data = _load()
        if not data:
            data = _DEFAULTS
            _save(data)
        for app, v in data.items():
            print(f"  {app}: {v['major']}.{v['minor']}.{v['patch']}")
        sys.exit(0)

    app = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "get"

    if action == "bump_patch":
        print(f"{app}: {bump_patch(app)}")
    elif action == "bump_minor":
        print(f"{app}: {bump_minor(app)}")
    elif action == "bump_major":
        print(f"{app}: {bump_major(app)}")
    else:
        print(f"{app}: {get_version(app)}")
