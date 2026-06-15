from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_project_env(project_dir: Path | None = None, override: bool = False) -> Path | None:
    root = project_dir or Path(__file__).resolve().parent
    env_path = root / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def get_tabpfn_token(project_dir: Path | None = None) -> str | None:
    load_project_env(project_dir=project_dir, override=False)
    return (
        os.getenv("TABPFN_API_TOKEN")
        or os.getenv("TABPFN_ACCESS_TOKEN")
        or os.getenv("TABPFN_TOKEN")
    )
