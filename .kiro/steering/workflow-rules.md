---
inclusion: auto
---

# Workflow Rules

These rules apply to ALL interactions in this workspace:

## Action Keywords

- **"do"** = Implement the code changes only. No git, no exe builds.
- **"go"** = Clean up the code, commit to develop, push to both branches (develop and main).
- **"go do"** = Clean up, commit, push to both branches, AND build new versioned exe files (new version numbers, not overwriting old ones).
- **No keyword** = Discuss, confirm, suggest only. Do NOT make code changes, do NOT commit, do NOT build.

## Exe Versioning

- Every "go do" creates NEW version-numbered exe files (e.g., v2.6.0, v2.7.0).
- Old versions stay in dist/ — never overwrite them.
- Both `WistDiscovery.exe` and `SudaneseWist_vX.Y.Z.exe` are built.
- A new `.spec` file is created for each new gameplay version.

## Git Rules

- Always push to BOTH branches: develop and main.
- Use short descriptive commit messages.

## File Locations

- The user runs exe files from the `dist/` folder.
- Model and cache files are relative to exe location: `dist/agents/wist_discovery/`.
- When clearing caches, clear inside `dist/` folder (not project root agents folder).
