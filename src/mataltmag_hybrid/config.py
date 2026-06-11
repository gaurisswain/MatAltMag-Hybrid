from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class GateError(RuntimeError):
    """Raised when a phase cannot proceed because required external inputs are missing."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        value = self.values[key]
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise GateError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise GateError(f"Config must be a mapping: {path}")
    return data


def load_paths(config: str | Path = "configs/paths.yaml") -> ProjectPaths:
    values = load_yaml(config)
    root_value = values.get("project_root", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = repo_root() / root
    return ProjectPaths(root=root.resolve(), values=values)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def require_files(paths: list[Path], purpose: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise GateError(f"{purpose} requires missing file(s): {', '.join(missing)}")

