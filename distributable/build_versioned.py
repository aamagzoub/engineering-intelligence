"""
Versioned build script for Telecom-Native-Intelligence executables.

Usage:
    python distributable/build_versioned.py WistDiscovery
    python distributable/build_versioned.py SudaneseWist_PyGame
    python distributable/build_versioned.py all

Each build:
1. Bumps the patch version automatically
2. Generates a Windows version-info file (.rc)
3. Builds the .exe with PyInstaller including version metadata
4. Renames output to include version: e.g., WistDiscovery_v2.0.1.exe
5. Keeps a copy as the plain name too (for shortcuts/launchers)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from version import bump_patch, get_version_tuple


# ---------------------------------------------------------------------------
# Build configurations
# ---------------------------------------------------------------------------

BUILDS = {
    "WistDiscovery": {
        "script": "gui_wist_discovery\\main.py",
        "windowed": True,
        "add_data": [
            "agents;agents",
            "environments;environments",
            "intelligence;intelligence",
            "tibrain;tibrain",
            "gui_wist_discovery;gui_wist_discovery",
            "gui_wist;gui_wist",
        ],
        "hidden_imports": [
            "pygame",
            "tibrain", "tibrain.agent", "tibrain.q_learning", "tibrain.q_table",
            "tibrain.policy", "tibrain.replay_buffer", "tibrain.neural_net",
            "tibrain.mcts", "tibrain.reward", "tibrain.evaluation",
            "tibrain.persistence", "tibrain.training",
            "tibrain.discovery", "tibrain.discovery.discovery_engine",
            "tibrain.discovery.pattern",
            "agents.wist_discovery.discovery_agent",
            "agents.wist_discovery.neural_net",
            "agents.wist_discovery.mcts",
            "environments.wist", "environments.wist.environment",
            "environments.wist.round", "environments.wist.rules",
            "environments.wist.scoring", "environments.wist.setup",
            "environments.wist.tasmiya_engine", "environments.wist.trick",
            "intelligence.core", "intelligence.core.cards",
            "gui_wist_discovery.game_engine", "gui_wist_discovery.training",
            "gui_wist_discovery.milestones", "gui_wist_discovery.insights",
            "gui_wist_discovery.renderer", "gui_wist_discovery.constants",
        ],
        "description": "Wist Discovery - AI Learning Watcher",
        "company": "Telecom-Native-Intelligence",
    },
    "SudaneseWist_PyGame": {
        "script": "gui_wist\\main.py",
        "windowed": True,
        "add_data": [
            "agents;agents",
            "environments;environments",
            "intelligence;intelligence",
            "tibrain;tibrain",
            "gui_wist;gui_wist",
        ],
        "hidden_imports": [
            "pygame",
            "tibrain", "tibrain.agent", "tibrain.q_learning", "tibrain.q_table",
            "tibrain.policy", "tibrain.replay_buffer", "tibrain.neural_net",
            "tibrain.mcts", "tibrain.reward", "tibrain.evaluation",
            "tibrain.persistence", "tibrain.training",
            "tibrain.discovery", "tibrain.discovery.discovery_engine",
            "tibrain.discovery.pattern",
            "agents.wist_rule_based.rule_based_agent",
            "agents.wist_learning.learning_agent",
            "environments.wist",
            "intelligence.core",
        ],
        "description": "Sudanese Wist - Card Game",
        "company": "Telecom-Native-Intelligence",
    },
}


# ---------------------------------------------------------------------------
# Version info file generation (Windows .rc format for PyInstaller)
# ---------------------------------------------------------------------------

VERSION_INFO_TEMPLATE = """
# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({major}, {minor}, {patch}, 0),
    prodvers=({major}, {minor}, {patch}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'{company}'),
            StringStruct(u'FileDescription', u'{description}'),
            StringStruct(u'FileVersion', u'{version_str}'),
            StringStruct(u'InternalName', u'{app_name}'),
            StringStruct(u'OriginalFilename', u'{app_name}.exe'),
            StringStruct(u'ProductName', u'{product_name}'),
            StringStruct(u'ProductVersion', u'{version_str}'),
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""


def generate_version_file(app_name: str, config: dict, version_tuple: tuple) -> Path:
    """Generate a PyInstaller version-info file and return its path."""
    major, minor, patch = version_tuple
    version_str = f"{major}.{minor}.{patch}"

    content = VERSION_INFO_TEMPLATE.format(
        major=major,
        minor=minor,
        patch=patch,
        company=config["company"],
        description=config["description"],
        version_str=version_str,
        app_name=app_name,
        product_name=config["description"],
    )

    version_file = PROJECT_ROOT / "build" / f"{app_name}_version.rc"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(content)
    return version_file


# ---------------------------------------------------------------------------
# Build execution
# ---------------------------------------------------------------------------

def build_app(app_name: str) -> bool:
    """Build a single application with version bumping."""
    if app_name not in BUILDS:
        print(f"ERROR: Unknown app '{app_name}'. Available: {list(BUILDS.keys())}")
        return False

    config = BUILDS[app_name]

    # Bump patch version
    new_version = bump_patch(app_name)
    version_tuple = get_version_tuple(app_name)
    print(f"\n{'='*60}")
    print(f"  Building {app_name} v{new_version}")
    print(f"{'='*60}\n")

    # Generate version info file
    version_file = generate_version_file(app_name, config, version_tuple)
    print(f"  Version info: {version_file}")

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconfirm",
        f"--name={app_name}",
        f"--version-file={version_file}",
    ]

    if config["windowed"]:
        cmd.append("--windowed")

    for data in config["add_data"]:
        cmd.extend(["--add-data", data])

    for imp in config["hidden_imports"]:
        cmd.extend(["--hidden-import", imp])

    cmd.append(config["script"])

    # Run PyInstaller
    print(f"  Running PyInstaller...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  BUILD FAILED!")
        print(result.stderr[-2000:] if result.stderr else "No error output")
        return False

    # Check output exists
    exe_path = PROJECT_ROOT / "dist" / f"{app_name}.exe"
    if not exe_path.exists():
        print(f"  BUILD FAILED - {exe_path} not found")
        return False

    # Create versioned copy
    versioned_name = f"{app_name}_v{new_version}.exe"
    versioned_path = PROJECT_ROOT / "dist" / versioned_name
    shutil.copy2(exe_path, versioned_path)

    file_size_mb = exe_path.stat().st_size / (1024 * 1024)

    print(f"\n  SUCCESS!")
    print(f"  Output:    dist/{app_name}.exe")
    print(f"  Versioned: dist/{versioned_name}")
    print(f"  Size:      {file_size_mb:.1f} MB")
    print(f"  Version:   {new_version}")
    print(f"{'='*60}\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python distributable/build_versioned.py <app_name|all>")
        print(f"\nAvailable apps: {', '.join(BUILDS.keys())}")
        print("\nExamples:")
        print("  python distributable/build_versioned.py WistDiscovery")
        print("  python distributable/build_versioned.py SudaneseWist_PyGame")
        print("  python distributable/build_versioned.py all")
        sys.exit(1)

    target = sys.argv[1]

    if target == "all":
        results = {}
        for app_name in BUILDS:
            results[app_name] = build_app(app_name)
        print("\n" + "=" * 60)
        print("  BUILD SUMMARY")
        print("=" * 60)
        for app, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  {app}: {status}")
        print("=" * 60)
        sys.exit(0 if all(results.values()) else 1)
    else:
        success = build_app(target)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
