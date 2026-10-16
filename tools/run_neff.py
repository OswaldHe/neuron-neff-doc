#!/usr/bin/env python3
"""Run an edited NEFF through a -full folder's own frozen validator, without touching that folder.

The -full validators (`inference.py`) compile `source.py` through torch_xla into a compile cache
at build/cache/<sha256(source.py)[:16]>/neuronxcc-*/MODULE_*/model.neff, then load that NEFF,
run it on 4 ranks, compare rank 0's output with the golden, and profile it. torch_xla takes the
NEFF from the cache whenever the cache entry exists ("Using a cached neff"), so replacing
model.neff in a cache entry makes the validator run the replacement instead.

This script does that in a sandbox:

  1. make <sandbox>/ with copies of inference.py and source.py and symlinks to everything
     else the validator reads (tensors/, reference/, module/, submodule/, ...);
  2. copy the original cache entry for source.py into <sandbox>/build/cache/<digest>/;
  3. overwrite its model.neff with the edited NEFF;
  4. run `NEURON_RT_NUM_CORES=4 torchrun --nproc_per_node=4 inference.py` in the sandbox and
     report the validator's ##autohelix[...] results.

Usage (workspace/.venv):
    run_neff.py <full-dir> <edited.neff> [--sandbox DIR] [--timeout SEC] [--exec-timeout SEC]

The original NEFF must be in the full folder's cache (run the validator there once, or use
gen-full-neffs.sh). Exit code is 0 when the validator reports passed=1.
"""

import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

WS = "/home/ubuntu/workspace"
VENV_BIN = os.path.join(WS, ".venv", "bin")


def cache_entry(full_dir):
    digest = hashlib.sha256(open(os.path.join(full_dir, "source.py"), "rb").read()).hexdigest()[:16]
    neffs = glob.glob(os.path.join(full_dir, "build", "cache", digest, "**", "model.neff"),
                      recursive=True)
    if len(neffs) != 1:
        raise SystemExit(f"expected one cached model.neff for source.py digest {digest} in "
                         f"{full_dir}/build/cache, found {len(neffs)}; run the validator there first")
    module_dir = os.path.dirname(neffs[0])
    return digest, module_dir


def make_sandbox(full_dir, sandbox, digest, module_dir, neff):
    if os.path.exists(sandbox):
        shutil.rmtree(sandbox)
    os.makedirs(sandbox)
    for name in os.listdir(full_dir):
        src = os.path.join(full_dir, name)
        if name in ("build", ".git", ".autohelix", "__pycache__") or name.endswith(".neff"):
            continue
        dst = os.path.join(sandbox, name)
        if name in ("inference.py", "source.py"):
            shutil.copyfile(src, dst)       # the validator resolves paths from its own file
        else:
            os.symlink(src, dst)
    rel = os.path.relpath(module_dir, os.path.join(full_dir, "build", "cache", digest))
    new_module = os.path.join(sandbox, "build", "cache", digest, rel)
    shutil.copytree(module_dir, new_module)
    shutil.copyfile(neff, os.path.join(new_module, "model.neff"))
    return new_module


def run(sandbox, timeout, exec_timeout=None):
    env = dict(os.environ)
    if exec_timeout:
        # the validator only sets NEURON_RT_EXEC_TIMEOUT if it is unset, so this value wins
        env["NEURON_RT_EXEC_TIMEOUT"] = str(int(exec_timeout))
    env["PATH"] = VENV_BIN + os.pathsep + env.get("PATH", "")
    env["NEURON_RT_NUM_CORES"] = "4"
    t0 = time.time()
    p = subprocess.run([os.path.join(VENV_BIN, "torchrun"), "--nproc_per_node=4", "inference.py"],
                       cwd=sandbox, env=env, capture_output=True, text=True, timeout=timeout)
    log = p.stdout + "\n" + p.stderr
    open(os.path.join(sandbox, "run.log"), "w").write(log)
    res = dict(re.findall(r"##autohelix\[(\w+)=([^\]]*)\]", log))
    res["returncode"] = p.returncode
    res["wall_s"] = round(time.time() - t0, 1)
    res["used_cached_neff"] = "Using a cached neff" in log
    res["compiled"] = "Compiler status PASS" in log or "Compiling" in log
    return res, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("neff")
    ap.add_argument("--sandbox")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--exec-timeout", type=float,
                    help="NEURON_RT_EXEC_TIMEOUT in seconds (a hung NEFF fails after this)")
    a = ap.parse_args()
    full = os.path.abspath(a.full_dir)
    neff = os.path.abspath(a.neff)
    sandbox = os.path.abspath(a.sandbox or os.path.join(
        WS, "neff_doc", "artifacts", "sandbox", os.path.basename(os.path.dirname(full)) + "-"
        + time.strftime("%Y%m%d-%H%M%S")))
    digest, module_dir = cache_entry(full)
    make_sandbox(full, sandbox, digest, module_dir, neff)
    res, log = run(sandbox, a.timeout, a.exec_timeout)
    for k in sorted(res):
        print(f"{k} = {res[k]}")
    print(f"sandbox = {sandbox}")
    if res.get("passed") != "1":
        tail = "\n".join(l for l in log.splitlines() if l.strip())[-3000:]
        print("---- log tail ----\n" + tail)
    return 0 if res.get("passed") == "1" else 1


if __name__ == "__main__":
    sys.exit(main())
