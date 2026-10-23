#!/usr/bin/env bash
# Rebuild every figure of BLOG.md from the NEFF/NTFF pairs in
# experiments/gdn_block_tkg_b1_h128_vh4_lnc1/blog_profiles/.
#   PY_TOOLS: a python with pandas + pyarrow (the workspace .venv works)
#   PY_PLOT:  a python with matplotlib >= 3.9 + pandas
# Re-measuring latency on a device is separate: measure_interleaved.py writes the results.json
# that export_data.py turns into figs/data/gdn_latency.json; without it the committed one is used.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PROF=$ROOT/experiments/gdn_block_tkg_b1_h128_vh4_lnc1/blog_profiles
PY_TOOLS=${PY_TOOLS:-python3}
PY_PLOT=${PY_PLOT:-python3}
WORK=$(mktemp -d)
for i in 0 1 2 3 4 5 6 7; do
  "$PY_TOOLS" "$ROOT/tools/neff.py" unpack "$PROF/iter$i.neff" "$WORK/unp/iter$i" > /dev/null
  neuron-explorer view -n "$PROF/iter$i.neff" -s "$PROF/iter$i.ntff" --output-format=parquet \
    --output-file="$WORK/pq/iter$i" --ingest-only > /dev/null
done
"$PY_TOOLS" "$HERE/export_data.py" "$WORK/unp" "$WORK/pq" ${RESULTS_JSON:-}
cd "$HERE"
"$PY_PLOT" fig_background.py
"$PY_PLOT" fig_data.py
"$PY_PLOT" fig_steps.py
rm -rf "$WORK"
