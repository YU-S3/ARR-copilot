from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="下载 TableGPT2-7B 到本地目录。")
    parser.add_argument(
        "--repo-id",
        default="tablegpt/TableGPT2-7B",
        help="Hugging Face 模型仓库名。",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=project_dir / "models" / "TableGPT2-7B",
        help="模型下载目录。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="可选 Hugging Face Token。",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="模型下载镜像端点，默认使用 hf-mirror。",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "snapshot", "git-lfs", "direct-resolve"],
        default="auto",
        help="下载方式：优先 snapshot，失败时可回退 direct-resolve 或 git-lfs。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="并发下载线程数，默认 4。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="是否强制重新下载。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查远端文件列表，不实际下载。",
    )
    return parser.parse_args()


def try_snapshot_download(args: argparse.Namespace) -> str | list[object]:
    return snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.local_dir),
        token=args.token,
        endpoint=args.endpoint,
        max_workers=args.max_workers,
        force_download=args.force_download,
        dry_run=args.dry_run,
    )


def git_lfs_download(args: argparse.Namespace) -> str:
    local_dir = args.local_dir.resolve()
    if shutil.which("git") is None:
        raise RuntimeError("未找到 git，无法使用 git-lfs 下载。")
    lfs_check = subprocess.run(
        ["git", "lfs", "version"],
        cwd=str(local_dir.parent),
        capture_output=True,
        text=True,
    )
    if lfs_check.returncode != 0:
        raise RuntimeError("未找到 git-lfs，无法使用 git-lfs 下载。")

    repo_url = f"{args.endpoint.rstrip('/')}/{args.repo_id}"
    if not repo_url.endswith(".git"):
        repo_url += ".git"

    if args.dry_run:
        subprocess.run(
            ["git", "ls-remote", repo_url],
            check=True,
            cwd=str(local_dir.parent),
        )
        return str(local_dir)

    git_dir = local_dir / ".git"
    if git_dir.exists():
        subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(local_dir))
        return str(local_dir)

    if local_dir.exists() and any(local_dir.iterdir()):
        shutil.rmtree(local_dir)

    subprocess.run(
        ["git", "clone", repo_url, str(local_dir)],
        check=True,
        cwd=str(local_dir.parent),
    )
    subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(local_dir))
    return str(local_dir)


def list_required_files(local_dir: Path) -> list[str]:
    base_files = [
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "README.md",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "vocab.json",
        "zero_to_fp32.py",
    ]
    required = list(base_files)
    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        shard_files = list(OrderedDict.fromkeys(data.get("weight_map", {}).values()))
        required.extend(shard_files)
    return required


def download_one_file(local_dir: Path, endpoint: str, repo_id: str, filename: str, dry_run: bool) -> None:
    target_path = local_dir / filename
    if dry_run:
        subprocess.run(
            [
                "curl.exe",
                "-I",
                "-L",
                f"{endpoint.rstrip('/')}/{repo_id}/resolve/main/{filename}",
            ],
            check=True,
            cwd=str(local_dir),
        )
        return

    if target_path.exists() and target_path.stat().st_size > 0:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl.exe",
            "-L",
            "-C",
            "-",
            "-o",
            str(target_path),
            f"{endpoint.rstrip('/')}/{repo_id}/resolve/main/{filename}",
        ],
        check=True,
        cwd=str(local_dir),
    )


def direct_resolve_download(args: argparse.Namespace) -> str:
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl.exe") is None:
        raise RuntimeError("未找到 curl.exe，无法使用 direct-resolve 下载。")

    base_files = [
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "README.md",
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "vocab.json",
        "zero_to_fp32.py",
    ]
    for filename in base_files:
        download_one_file(local_dir, args.endpoint, args.repo_id, filename, args.dry_run)

    # 下载完 index 后重新读取，确保四个权重分片被纳入队列。
    required_files = list_required_files(local_dir)
    shard_files = [filename for filename in required_files if filename.endswith(".safetensors")]
    for filename in shard_files:
        download_one_file(local_dir, args.endpoint, args.repo_id, filename, args.dry_run)
    return str(local_dir)


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    args.local_dir.mkdir(parents=True, exist_ok=True)
    download_method = args.method
    error_message = None
    if args.method in {"auto", "snapshot"}:
        try:
            downloaded_path = try_snapshot_download(args)
            download_method = "snapshot"
        except Exception as exc:
            if args.method == "snapshot":
                raise
            error_message = str(exc)
            try:
                downloaded_path = direct_resolve_download(args)
                download_method = "direct-resolve"
            except Exception as direct_exc:
                error_message = f"{error_message}\n[direct-resolve fallback] {direct_exc}"
                downloaded_path = git_lfs_download(args)
                download_method = "git-lfs"
    elif args.method == "direct-resolve":
        downloaded_path = direct_resolve_download(args)
        download_method = "direct-resolve"
    else:
        downloaded_path = git_lfs_download(args)
        download_method = "git-lfs"

    manifest = {
        "repo_id": args.repo_id,
        "endpoint": args.endpoint,
        "method": download_method,
        "max_workers": args.max_workers,
        "dry_run": args.dry_run,
        "downloaded_path": str(downloaded_path),
        "snapshot_error": error_message,
    }
    (args.local_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
