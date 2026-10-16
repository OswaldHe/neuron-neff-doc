#!/usr/bin/env bash
# Generate the NEFF of every optimization-runs-*/*-full/source.py by running that folder's own
# frozen validator (the gate's exact invocation), then copy the NEFF it compiled for the
# *current* source.py -- found under build/cache/<sha256(source.py)[:16]>/ -- to
# optimization-runs-*/<full-dir>.neff.
#
# Usage: gen-full-neffs.sh <full-dir> [<full-dir> ...]
set -u
source /home/ubuntu/workspace/.venv/bin/activate
LOGS=/home/ubuntu/workspace/neff_doc/artifacts/neff_generation_logs
mkdir -p "$LOGS"

for full in "$@"; do
  full=$(realpath "$full")
  run=$(dirname "$full")
  name=$(basename "$full")
  tag="$(basename "$run")__$name"
  log="$LOGS/$tag.log"
  digest=$(sha256sum "$full/source.py" | cut -c1-16)
  echo "[$(date +%T)] START $tag digest=$digest"

  t0=$(date +%s)
  (cd "$full" && NEURON_RT_NUM_CORES=4 timeout 3h \
      torchrun --nproc_per_node=4 inference.py) >"$log" 2>&1
  rc=$?
  t1=$(date +%s)

  mapfile -t neffs < <(find "$full/build/cache/$digest" -name '*.neff' 2>/dev/null)
  passed=$(grep -oE '##autohelix\[passed=[01]\]' "$log" | tail -1)
  lat=$(grep -oE '##autohelix\[latency_ms=[^]]*\]' "$log" | tail -1)
  if [ "${#neffs[@]}" -eq 1 ]; then
    cp "${neffs[0]}" "$run/$name.neff"
    echo "[$(date +%T)] DONE  $tag rc=$rc $((t1 - t0))s $passed $lat -> $run/$name.neff" \
         "($(stat -c %s "$run/$name.neff") bytes, from ${neffs[0]#$full/})"
  else
    echo "[$(date +%T)] FAIL  $tag rc=$rc $((t1 - t0))s: ${#neffs[@]} neff(s) under" \
         "build/cache/$digest; see $log"
  fi
done
