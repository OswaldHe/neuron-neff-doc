#!/usr/bin/env bash
# Turn each studied NEFF's rank-0 profile into parquet tables under
# neff_doc/artifacts/profiles/<tag>/rank0/. The Instruction table is what links a NEFF
# instruction (engine, compiler_pc) to the profiler's own disassembly text.
#
# A NEFF with no profile yet is captured first, the same way the -full validators capture:
# one 4-rank collective capture on an idle device.
set -u
ROOT=/home/ubuntu/workspace
NX=/opt/aws/neuron/bin/neuron-explorer
OUT=$ROOT/neff_doc/artifacts/profiles

while read -r neff; do
  tag=$(dirname "$neff" | sed 's/^optimization-runs-//')
  build=$ROOT/${neff%.neff}/build
  ntff=$build/profile_rank_0.ntff
  [ -f "$ntff" ] || ntff=$build/profile_cap0_rank_0.ntff
  if [ ! -f "$ntff" ]; then
    mkdir -p "$build/neffdoc-capture"
    cp "$ROOT/$neff" "$build/neffdoc-capture/profiled.neff"
    (cd "$build/neffdoc-capture" && NEURON_RT_NUM_CORES=4 "$NX" capture -n profiled.neff \
        --io-from=runtime --collectives-worker-count 4 --collectives-workers-per-node 4 \
        --collectives-worker-start-id 0 --collectives-profile-id all -s profile.ntff) \
        > "$OUT/capture_$tag.log" 2>&1
    ntff=$build/neffdoc-capture/profile_rank_0.ntff
  fi
  rm -rf "$OUT/$tag/rank0"
  mkdir -p "$OUT/$tag"
  start=$(date +%s)
  "$NX" view -n "$ROOT/$neff" -s "$ntff" --output-format=parquet \
      --output-file="$OUT/$tag/rank0" --ingest-only > "$OUT/$tag/ingest.log" 2>&1
  echo "[$(date +%T)] $tag rc=$? $(( $(date +%s) - start ))s ntff=${ntff#$ROOT/}"
done < "$ROOT/neff_doc/artifacts/neff_list.txt"
