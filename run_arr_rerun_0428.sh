#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash run_arr_rerun_0428.sh [smoke|full]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="$ROOT_DIR/data_0428.xlsx"
OUTPUT_ROOT="$ROOT_DIR/rerun_0428_outputs/$MODE"
MANIFEST="$OUTPUT_ROOT/rerun_manifest.json"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  conda_candidates=(
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/d/Download/anaconda/etc/profile.d/conda.sh" \
    "D:/Download/anaconda/etc/profile.d/conda.sh"
  )
  if [[ -f "/mnt/d/Download/anaconda/etc/profile.d/conda.sh" ]] && command -v cygpath >/dev/null 2>&1; then
    conda_candidates+=("/mnt/d/Download/anaconda/etc/profile.d/conda.sh")
  fi
  for conda_sh in "${conda_candidates[@]}"; do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Run this script from a shell where conda is initialized, then retry:" >&2
  echo "  conda activate arr_rf" >&2
  echo "  bash run_arr_rerun_0428.sh $MODE" >&2
  exit 127
fi
conda activate arr_rf

mkdir -p "$OUTPUT_ROOT/logs"

python - "$MANIFEST" "$MODE" "$INPUT_FILE" "$OUTPUT_ROOT" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
mode = sys.argv[2]
input_file = sys.argv[3]
output_root = Path(sys.argv[4])
steps = [
    ("01_ensemble", "基础模型重跑"),
    ("02_scm_v2", "SCM-v2 主线"),
    ("03_scm_v3", "SCM-v3 最优候选"),
    ("04_v31", "V3.1 校准"),
]
now = datetime.now().isoformat(timespec="seconds")
manifest = {
    "mode": mode,
    "status": "running",
    "input_file": input_file,
    "output_root": str(output_root),
    "started_at": now,
    "updated_at": now,
    "steps": [
        {
            "id": step_id,
            "label": label,
            "status": "pending",
            "output_dir": str(output_root / step_id),
            "log_file": str(output_root / "logs" / f"{step_id}.log"),
            "started_at": None,
            "ended_at": None,
        }
        for step_id, label in steps
    ],
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

update_manifest() {
  local step_id="$1"
  local status="$2"
  python - "$MANIFEST" "$step_id" "$status" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
step_id = sys.argv[2]
status = sys.argv[3]
now = datetime.now().isoformat(timespec="seconds")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["updated_at"] = now
for step in manifest["steps"]:
    if step["id"] == step_id:
        step["status"] = status
        if status == "running":
            step["started_at"] = now
        if status in {"done", "failed"}:
            step["ended_at"] = now
        break
if status == "failed":
    manifest["status"] = "failed"
elif all(step["status"] == "done" for step in manifest["steps"]):
    manifest["status"] = "done"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_step() {
  local step_id="$1"
  shift
  local log_file="$OUTPUT_ROOT/logs/$step_id.log"
  echo "[$(date '+%F %T')] START $step_id" | tee "$log_file"
  update_manifest "$step_id" "running"
  if "$@" >>"$log_file" 2>&1; then
    update_manifest "$step_id" "done"
    echo "[$(date '+%F %T')] DONE  $step_id" | tee -a "$log_file"
  else
    update_manifest "$step_id" "failed"
    echo "[$(date '+%F %T')] FAILED $step_id; see $log_file" | tee -a "$log_file" >&2
    exit 1
  fi
}

if [[ "$MODE" == "smoke" ]]; then
  ENSEMBLE_ARGS=(--n-trials 1 --n-splits 2)
  SCM_V2_ARGS=(--smoke-test --phase1-only)
  SCM_V3_ARGS=(--smoke-test --best-only)
  V31_ARGS=(--smoke-test)
else
  ENSEMBLE_ARGS=(--n-trials 6 --n-splits 4)
  SCM_V2_ARGS=()
  SCM_V3_ARGS=(--best-only)
  V31_ARGS=()
fi

run_step "01_ensemble" python "$ROOT_DIR/multiclass_ensemble_experiment.py" \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_ROOT/01_ensemble" \
  "${ENSEMBLE_ARGS[@]}"

run_step "02_scm_v2" python "$ROOT_DIR/frontier_scm_v2_experiment.py" \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_ROOT/02_scm_v2" \
  --skip-tabpfn \
  "${SCM_V2_ARGS[@]}"

run_step "03_scm_v3" python "$ROOT_DIR/frontier_scm_v3_experiment.py" \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_ROOT/03_scm_v3" \
  --skip-tabpfn \
  "${SCM_V3_ARGS[@]}"

run_step "04_v31" python "$ROOT_DIR/scm_v31_experiment.py" \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_ROOT/04_v31" \
  --skip-tabddpm \
  "${V31_ARGS[@]}"

echo "All rerun steps completed. Output: $OUTPUT_ROOT"
