#!/usr/bin/env bash
# Paper parity check: rerun the paper's main Eindhoven configuration (TGF 2026, run of 30 Sept)
# on N seeded test tracks and compare every per-event row, and every quantity fitted on the
# calibration tracks, with the paper's final run. --test-subset leaves calibration and every
# fit unchanged, so the rows for those tracks must be identical.
#   tests/parity/run_parity.sh [N=5]
#
# Inputs, overridable by environment:
#   REACT_REPRO_DIR  clone of github.com/calvin-breseman/react-reproducibility: the reference
#                    rows (subsets up to 20 tracks) and the paper's checkpoints and IMM fit
#   REACT_CSV        the 20,000-track Eindhoven CSV (scripts/export_eindhoven_csv.py)
#   TPP_DIR          Trajectron++ at commit 1031c7bd with the reproducibility repo's patch
#   PYTHON           interpreter (default: the project venv)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RM="$(cd "$HERE/../.." && pwd)"
REPRO="${REACT_REPRO_DIR:-$HOME/react-reproducibility}"
CSV="${REACT_CSV:-$RM/.data/eindhoven_20000tracks.csv}"
TPP="${TPP_DIR:-$HOME/Trajectron-plus-plus}"
PY="${PYTHON:-$RM/.venv/bin/python}"
N="${1:-5}"
REF="$REPRO/reference/eindhoven_floorA"
A="$REPRO/artifacts/eindhoven"
for p in "$REF/subset_records.csv" "$CSV" "$TPP" "$A/imm_fit.json"; do
  [ -e "$p" ] || { echo "parity: missing $p" >&2; exit 2; }
done
OUT="${PARITY_OUT:-$(mktemp -d)}"
cd "$RM/experiments"
"$PY" exp10_information_react.py --csv "$CSV" --seed 0 --jobs 8 --tests exact --no-still-cap \
  --lead1-levels 4.2 --entropy-margin-levels 4.2 --distortions 0.1 0.3 --no-tpp-gaussian \
  --tpp-dir "$TPP" --tpp-model "$A/trajectron_run1" --tracks 1200 \
  --gru-dir "$A/gru_10hz" --gru-at5-dir "$A/gru_10hz_at5" --imm-params "$A/imm_fit.json" \
  --noise-floor A --levels 3 4.2 6 --test-subset "$N" --out "$OUT" > "$OUT/run.log" 2>&1 \
  || { echo "parity: run failed, log in $OUT/run.log" >&2; tail -20 "$OUT/run.log" >&2; exit 1; }
"$PY" "$HERE/compare.py" "$OUT" "$REF/subset_records.csv" \
  "$REF/summary_seed0_n1200_vw1_uncapped.json"
