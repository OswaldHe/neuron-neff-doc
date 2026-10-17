#!/usr/bin/env python3
"""The constraint: run `kernel.neff` on the device and check it still computes the right answer.

This is the only thing standing between an edit and a wrong kernel, so it checks the whole
chain and prints PASS/FAIL per claim:

  1. `kernel.neff` is a well-formed NEFF      header size and both MD5 copies match the payload
  2. its interface and shape are unchanged    same input tensor names, same number of outputs,
                                              still LNC=1 and single-rank (tools/single_core.py)
  3. the device loaded and ran it             capture exit code, no hang, no out-of-bounds
  4. every output came back finite            no NaN, no Inf
  5. every output matches the golden          `max_abs_err <= 8e-4` (`max_abs_err_limit` in
                                              bench/manifest.json), and also within the upstream
                                              test's much looser atol/rtol
  6. (--runs N) repeated runs agree bit for bit -- catches a race a single run would miss

Exit code 0 means every claim held; that is what the loop treats as "the edit is allowed".

Usage:
    python tools/verify.py                 # one run
    python tools/verify.py --runs 3        # also require run-to-run determinism
    python tools/verify.py --keep          # keep the device output tensors
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import device as D  # noqa: E402
import neffio  # noqa: E402
import single_core  # noqa: E402

WORK = os.path.join(D.ROOT, ".verify")


def compare(dev: np.ndarray, gold: np.ndarray, atol: float, rtol: float) -> dict:
    d, g = np.asarray(dev, np.float32).ravel(), np.asarray(gold, np.float32).ravel()
    err = np.abs(d - g)
    tol = atol + rtol * np.abs(g)
    nd, ng = np.linalg.norm(d), np.linalg.norm(g)
    return {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "max_rel_err": float((err / np.maximum(np.abs(g), 1e-6)).max()),
            "cosine": float(np.dot(d, g) / (nd * ng)) if nd and ng else 0.0,
            "within_tol_fraction": float((err <= tol).mean()),
            "allclose": bool(np.all(err <= tol)),
            "nan_or_inf": int((~np.isfinite(d)).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=1,
                    help="device runs; more than one also requires bit-identical outputs")
    ap.add_argument("--timeout", type=int, default=600, help="per-run timeout in seconds")
    ap.add_argument("--keep", action="store_true", help="keep .verify/ with the output tensors")
    a = ap.parse_args()

    m = D.manifest()
    report = {"case": m["case"], "runs": a.runs, "checks": [], "pass": True}

    def check(ok, claim, detail=""):
        report["checks"].append({"pass": bool(ok), "claim": claim, "detail": detail})
        report["pass"] &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {claim}" + (f"  ({detail})" if detail else ""))
        return ok

    print(f"verifying kernel.neff against {m['case']}  {m['params']}")
    print(f"  sha256 {neffio.sha256(D.NEFF)}  ({os.path.getsize(D.NEFF)} bytes)")

    problems = neffio.header_problems(D.NEFF)
    if not check(not problems, "kernel.neff is a well-formed NEFF", "; ".join(problems)):
        D.save_json(os.path.join(D.ROOT, "verify_latest.json"), report)
        return 1

    sig = neffio.io_signature(D.NEFF)
    structural = single_core.problems(D.NEFF, m)
    check(not structural,
          "same interface, still LNC=1 and single-rank (see tools/single_core.py)",
          "; ".join(structural))
    report["io_signature"] = sig
    if not report["pass"]:
        D.save_json(os.path.join(D.ROOT, "verify_latest.json"), report)
        return 1

    runs = []
    for r in range(a.runs):
        work = os.path.join(WORK, f"run{r}")
        shutil.rmtree(work, ignore_errors=True)
        res = D.capture(work, m, save_outputs=True, timeout=a.timeout)
        how = f"TIMED OUT after {a.timeout}s" if res["timed_out"] else f"exit {res['returncode']}"
        detail = f"{how}, {res['wall_seconds']}s"
        if not check(res["returncode"] == 0, f"run {r}: the device ran the NEFF", detail):
            tail = (res["stderr"] or res["stdout"]).strip().splitlines()[-12:]
            print("       " + "\n       ".join(tail))
            report["capture"] = res
            D.save_json(os.path.join(D.ROOT, "verify_latest.json"), report)
            return 1
        try:
            runs.append(D.read_outputs(work, m))
        except RuntimeError as exc:
            check(False, f"run {r}: every output tensor came back", str(exc))
            D.save_json(os.path.join(D.ROOT, "verify_latest.json"), report)
            return 1

    if a.runs > 1:
        same = all(np.array_equal(np.asarray(runs[0][n]).view(np.uint8),
                                  np.asarray(runs[r][n]).view(np.uint8))
                   for n in runs[0] for r in range(1, a.runs))
        check(same, f"all {a.runs} runs produced bit-identical outputs")

    limit = m.get("max_abs_err_limit", 8e-4)
    report["max_abs_err_limit"] = limit
    report["outputs"] = {}
    for t, gold in D.golden(m).items():
        c = compare(runs[0][t], gold, m["atol"], m["rtol"])
        report["outputs"][t] = c
        check(c["nan_or_inf"] == 0, f"{t}: finite", f"{c['nan_or_inf']} NaN or Inf")
        # The binding numeric check. Much tighter than the upstream test's atol/rtol (5e-2),
        # which this kernel passes with room to spare: the point here is that an edit must not
        # change the numbers at all beyond what the unedited NEFF already differs by.
        check(c["max_abs_err"] <= limit, f"{t}: max_abs_err <= {limit:g}",
              f"max_abs_err {c['max_abs_err']:.3e} "
              f"({100 * c['max_abs_err'] / limit:.0f}% of the limit), "
              f"cosine {c['cosine']:.6f}")
        check(c["allclose"], f"{t}: also within the upstream atol={m['atol']:g} "
                             f"rtol={m['rtol']:g}",
              f"within tol {100 * c['within_tol_fraction']:.2f}%")

    D.save_json(os.path.join(D.ROOT, "verify_latest.json"), report)
    if not a.keep:
        shutil.rmtree(WORK, ignore_errors=True)
    print("VERIFIED" if report["pass"] else "FAILED -- this edit must not be kept")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
