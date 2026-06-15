from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="打包 TableGPT2 PA 实验工程，便于上传服务器。")
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=project_dir / "tablegpt2_pa_bundle.zip",
    )
    parser.add_argument(
        "--include-model-dir",
        type=Path,
        default=None,
        help="可选：将已下载的基础模型目录一并打包。",
    )
    parser.add_argument(
        "--include-output-dir",
        type=Path,
        default=project_dir / "tablegpt2_pa_outputs",
        help="可选：将实验输出目录一并打包。",
    )
    return parser.parse_args()


def add_path_to_zip(archive: zipfile.ZipFile, path: Path, base_dir: Path) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                archive.write(child, child.relative_to(base_dir))
    elif path.is_file():
        archive.write(path, path.relative_to(base_dir))


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    include_paths = [
        project_dir / "tablegpt2_pa",
        project_dir / "tap_gpt_pa_experiment_plan.md",
        project_dir / "environment.yml",
        project_dir / "environment_tablegpt2_pa.yml",
    ]

    requirements_path = project_dir / "requirements_tablegpt2_pa.txt"
    if requirements_path.exists():
        include_paths.append(requirements_path)
    if args.include_output_dir and args.include_output_dir.exists():
        include_paths.append(args.include_output_dir)
    if args.include_model_dir and args.include_model_dir.exists():
        include_paths.append(args.include_model_dir)

    args.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include_paths:
            add_path_to_zip(archive, path, project_dir)

    manifest = {
        "bundle_path": str(args.bundle_path),
        "included_paths": [str(path) for path in include_paths],
    }
    manifest_path = args.bundle_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
