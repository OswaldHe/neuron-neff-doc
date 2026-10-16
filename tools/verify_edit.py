#!/usr/bin/env python3
"""Run an edited NEFF and check the whole evidence chain that it, and not something else, ran.

"The edit works as intended" is four separate claims. This checks them one at a time and prints
PASS/FAIL per link, so a failure says which claim broke:

  1. the file is well formed        header size and both MD5 copies match the payload
  2. my file is the one that ran    the NEFF the validator loaded and profiled is byte-identical
                                    to the edited file; torch_xla did not recompile source.py
  3. the profile describes my file  the profile's NeffHeader uuid/hash equal the edited file's,
                                    and neuron-explorer accepted the NEFF/NTFF pair (it refuses
                                    a NEFF whose uuid differs from the one the NTFF recorded)
  4. the hardware ran the edit      the profiler's own disassembly of the slots named by
                                    --slots shows the edited operands
  5. the effect is as predicted     with --baseline, the device output is compared against a
                                    baseline tensor; with --expect, against a predicted tensor

Usage (workspace/.venv):
    verify_edit.py <full-dir> <edited.neff> [--slots sgNN:Engine:first-last]
                   [--baseline a.pt] [--expect b.pt] [--sandbox D] [--exec-timeout S]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import neff  # noqa: E402

WS = "/home/ubuntu/workspace"
NX = "/opt/aws/neuron/bin/neuron-explorer"
PROF_ENGINE = {"Activation": "Scalar", "DVE": "Vector", "PE": "Tensor", "Pool": "GpSimd",
               "SP": "Sync"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def say(ok, claim, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}" + (f"\n         {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("neff")
    ap.add_argument("--slots", help="sgNN:Engine:first-last, e.g. sg00:Pool:738-741")
    ap.add_argument("--baseline")
    ap.add_argument("--expect")
    ap.add_argument("--sandbox")
    ap.add_argument("--exec-timeout", default="300")
    a = ap.parse_args()
    full, edited = os.path.abspath(a.full_dir), os.path.abspath(a.neff)
    sandbox = os.path.abspath(a.sandbox or os.path.join(os.path.dirname(edited), "verify_sandbox"))
    ok = True

    print("1. the file is well formed")
    problems = neff.verify(edited)
    ok &= say(not problems, "header size and MD5 match the payload", "; ".join(problems))
    hdr = neff.read(edited)[0]

    print("\n2. my file is the one that ran")
    shutil.rmtree(sandbox, ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_neff.py"), full, edited,
                        "--sandbox", sandbox, "--exec-timeout", a.exec_timeout],
                       capture_output=True, text=True, timeout=7200)
    res = dict(re.findall(r"^(\w+) = (.*)$", r.stdout, re.M))
    open(os.path.join(sandbox, "verify_run.out"), "w").write(r.stdout + r.stderr)
    prof_neff = os.path.join(sandbox, "build", "profiled.neff")
    ok &= say(os.path.exists(prof_neff) and sha(prof_neff) == sha(edited),
              "the NEFF the validator loaded and profiled is byte-identical to the edited file",
              f"sha256 {sha(prof_neff)[:16] if os.path.exists(prof_neff) else 'missing'} vs "
              f"{sha(edited)[:16]}")
    ok &= say(res.get("used_cached_neff") == "True" and res.get("compiled") == "False",
              "the runtime took the NEFF from the cache and did not recompile source.py",
              f"used_cached_neff={res.get('used_cached_neff')} compiled={res.get('compiled')}")

    print("\n3. the profile describes my file")
    ntff = next((os.path.join(sandbox, "build", f) for f in sorted(os.listdir(
        os.path.join(sandbox, "build"))) if re.match(r"profile(_cap\d+)?_rank_0\.ntff", f)), None)
    prof = os.path.join(sandbox, "prof")
    shutil.rmtree(prof, ignore_errors=True)
    v = subprocess.run([NX, "view", "-n", prof_neff, "-s", ntff, "--output-format=parquet",
                        "--output-file=" + prof, "--ingest-only"], capture_output=True, text=True)
    ok &= say(v.returncode == 0 and os.path.exists(os.path.join(prof, "NeffHeader.parquet")),
              "neuron-explorer accepted this NEFF with this profile (it refuses a uuid mismatch)",
              (v.stdout + v.stderr)[-200:] if v.returncode else "")
    import pandas as pd
    nh = pd.read_parquet(os.path.join(prof, "NeffHeader.parquet")).iloc[0]
    ok &= say(nh["uuid"] == hdr["payload_md5"][1] and nh["hash"] == hdr["payload_md5"][0],
              "the profile's NeffHeader uuid and hash equal the edited file's header",
              f"profile uuid={nh['uuid']} file={hdr['payload_md5'][1]}")

    if a.slots:
        print("\n4. the hardware ran the edited instructions")
        sg, eng, rng = a.slots.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        df = pd.read_parquet(os.path.join(prof, "Instruction.parquet"),
                             columns=["engine", "pcore_idx", "compiler_pc", "opcode", "operands"])
        d = df[(df.engine == PROF_ENGINE[eng]) & (df.pcore_idx == int(sg[-2:]))]
        # the profiler's compiler_pc label can be one ahead of the slot index (chapter 11.4)
        d = d[d.compiler_pc.between(lo, hi + 1)].drop_duplicates("compiler_pc")
        say(len(d) > 0, f"{len(d)} executed instructions found for {a.slots}")
        for row in d.sort_values("compiler_pc").itertuples():
            print(f"         label {int(row.compiler_pc)} (slot {int(row.compiler_pc) - 1} or "
                  f"{int(row.compiler_pc)}): {row.opcode}: {(row.operands or '')[:150]}")

    out = os.path.join(sandbox, "build", "rank0_output.pt")
    if a.baseline or a.expect:
        print("\n5. the effect on the output")
        import torch
        dev = torch.load(out) if os.path.exists(out) else None
        ok &= say(dev is not None, "the run produced rank 0's output")
        if dev is not None and a.baseline:
            base = torch.load(a.baseline)
            same = bool(torch.equal(dev.reshape(base.shape), base))
            say(not same, "the output differs from the baseline (the edit had an effect)",
                f"rows differing: {int((dev.reshape(base.shape) != base).any(dim=-1).sum())}")
        if dev is not None and a.expect:
            exp = torch.load(a.expect)
            ok &= say(bool(torch.equal(dev.reshape(exp.shape), exp)),
                      "the output equals the prediction, bit for bit")

    print(f"\nvalidator: passed={res.get('passed')} max_abs_err={res.get('max_abs_err')} "
          f"latency_ms={res.get('latency_ms')}")
    print(f"sandbox: {sandbox}")
    print(f"\n{'ALL CHECKED CLAIMS PASS' if ok else 'SOME CLAIMS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
