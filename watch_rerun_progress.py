from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
RERUN_ROOT = PROJECT_DIR / "rerun_0428_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ARR rerun progress.")
    parser.add_argument("mode", nargs="?", choices=["smoke", "full"], help="Show smoke or full; default is latest.")
    parser.add_argument("--root", type=Path, help="Specific output directory containing rerun_manifest.json.")
    parser.add_argument("--tail", type=int, default=3, help="Recent log lines to show for each step.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None


def choose_manifest(mode: str | None, root: Path | None) -> Path | None:
    if root is not None:
        root = root if root.is_absolute() else PROJECT_DIR / root
        path = root / "rerun_manifest.json"
        return path if path.exists() else None

    if mode:
        candidates = []
        direct = RERUN_ROOT / mode / "rerun_manifest.json"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(RERUN_ROOT.glob(f"*/{mode}/rerun_manifest.json"))
        candidates = sorted(
            candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    candidates = sorted(
        RERUN_ROOT.glob("**/rerun_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def progress_snapshot(step: dict[str, Any]) -> tuple[str, str]:
    progress_path = Path(step["output_dir"]) / "progress.json"
    progress = load_json(progress_path)
    if not progress:
        return "-", "-"
    percent = progress.get("progress_percent", "-")
    stage = progress.get("current_stage") or progress.get("stage") or "-"
    if isinstance(percent, (int, float)):
        percent_text = f"{percent:.2f}%"
    else:
        percent_text = str(percent)
    return percent_text, str(stage)


def tail_lines(path: Path, count: int) -> list[str]:
    if count <= 0 or not path.exists():
        return []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    text = ""
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            candidate = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if candidate.count("\x00") <= max(1, len(candidate) // 20):
            text = candidate
            break
    if not text:
        text = raw.decode("utf-8", errors="replace").replace("\x00", "")
    lines = text.splitlines()
    return lines[-count:]


def main() -> None:
    args = parse_args()
    manifest_path = choose_manifest(args.mode, args.root)
    if manifest_path is None:
        print("No rerun_manifest.json found. Run a smoke or full script first.")
        return

    manifest = load_json(manifest_path)
    if not manifest:
        print(f"Could not read manifest: {manifest_path}")
        return

    print(f"Manifest: {manifest_path}")
    print(f"Mode: {manifest.get('mode')} | Status: {manifest.get('status')} | Updated: {manifest.get('updated_at')}")
    print(f"Input: {manifest.get('input_file')}")
    print()

    for step in manifest.get("steps", []):
        percent, stage = progress_snapshot(step)
        print(
            f"[{step.get('status', 'unknown')}] {step.get('id')} - {step.get('label')} | "
            f"progress={percent} | stage={stage}"
        )
        print(f"  output: {step.get('output_dir')}")
        print(f"  start: {step.get('started_at')} | end: {step.get('ended_at')}")
        recent = tail_lines(Path(step["log_file"]), args.tail)
        if recent:
            print("  recent log:")
            for line in recent:
                print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
