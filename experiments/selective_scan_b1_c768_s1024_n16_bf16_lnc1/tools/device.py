#!/usr/bin/env python3
"""Running `kernel.neff` on the device, and reading a profile of it.

Both the verifier and the profiler go through `neuron-explorer capture`, which loads a NEFF onto
a real NeuronCore and executes it. The same call can write the output tensors back to the host
(what the verifier needs) and capture a hardware profile (what the profiler needs).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURON_EXPLORER = "/opt/aws/neuron/bin/neuron-explorer"
NEFF = os.path.join(ROOT, "kernel.neff")
BENCH = os.path.join(ROOT, "bench")
MANIFEST = os.path.join(BENCH, "manifest.json")


def manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def to_dtype(name: str) -> np.dtype:
    """numpy dtype for a NKI dtype name; bfloat16 and the float8s come from ml_dtypes."""
    try:
        return np.dtype(name)
    except TypeError:
        import ml_dtypes

        return np.dtype(getattr(ml_dtypes, name))


def input_args(m: dict) -> list[str]:
    """The `<NAME> <FILE>` pairs `neuron-explorer capture` takes as its input map."""
    args = []
    for t in m["inputs"]:
        args += [t["name"], os.path.join(BENCH, t["file"])]
    return args


def capture(workdir: str, m: dict, *, neff: str = NEFF, ntff: str | None = None,
            num_exec: int = 1, profile_nth: int | None = None, save_outputs: bool = False,
            timeout: int = 600) -> dict:
    """Execute the NEFF once (or `num_exec` times) with the benchmark's inputs.

    A NEFF with broken synchronisation hangs instead of failing, so every call has a timeout;
    a timeout is reported like any other failure (`returncode` -9 / `timed_out`). The device is
    usable again on the next call.
    """
    os.makedirs(workdir, exist_ok=True)
    cmd = [NEURON_EXPLORER, "capture", "-n", os.path.abspath(neff)]
    cmd += ["-s", os.path.abspath(ntff)] if ntff else ["--disable-profile"]
    if num_exec != 1:
        cmd += ["--num-exec", str(num_exec)]
    if profile_nth is not None:
        cmd += ["--profile-nth-exec", str(profile_nth)]
    if save_outputs:
        cmd += ["--save-output-npy"]
    cmd += input_args(m)

    env = dict(os.environ)
    # The kernel is single-core; this host's runtime defaults to LNC=2 and a mismatch between
    # the NEFF and NEURON_LOGICAL_NC_CONFIG is a load-time error.
    env["NEURON_LOGICAL_NC_CONFIG"] = str(m.get("lnc", 1))

    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
                           env=env)
        rc, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as exc:
        rc, out, err, timed_out = -9, exc.stdout or "", exc.stderr or "", True
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    if isinstance(err, bytes):
        err = err.decode(errors="replace")
    return {"cmd": " ".join(cmd), "returncode": rc, "timed_out": timed_out,
            "stdout": out[-4000:], "stderr": err[-4000:],
            "wall_seconds": round(time.time() - t0, 2)}


def read_outputs(workdir: str, m: dict) -> dict[str, np.ndarray]:
    """Read back every output tensor the capture wrote.

    `--save-output-npy` writes `<name>.npy` where the npy format can hold the dtype and falls
    back to a raw `<name>.bin` where it cannot (bfloat16). The npy carries the runtime's flat
    shape, so the manifest's shape and dtype decide in both cases.
    """
    outs = {}
    for t in m["outputs"]:
        dt = to_dtype(t["dtype"])
        npy, bin_ = os.path.join(workdir, t["name"] + ".npy"), os.path.join(workdir,
                                                                           t["name"] + ".bin")
        if os.path.exists(npy):
            a = np.load(npy)
            if a.dtype.itemsize != dt.itemsize:
                raise RuntimeError(f"{npy}: {a.dtype} does not fit {t['dtype']}")
            a = a.view(dt)
        elif os.path.exists(bin_):
            a = np.fromfile(bin_, dtype=dt)
        else:
            raise RuntimeError(f"output {t['name']} was not written to {workdir}")
        want = int(np.prod(t["shape"]))
        if a.size != want:
            raise RuntimeError(f"{t['name']}: {a.size} elements, expected {want}")
        outs[t["name"]] = a.reshape(t["shape"])
    return outs


def golden(m: dict) -> dict[str, np.ndarray]:
    return {t["name"]: np.load(os.path.join(BENCH, t["golden"])) for t in m["outputs"]}


def find_ntff(stem: str) -> str | None:
    """`capture -s x.ntff --num-exec N` writes `x_exec_N.ntff`; with one execution, `x.ntff`."""
    if os.path.exists(stem + ".ntff"):
        return stem + ".ntff"
    found = sorted(glob.glob(stem + "_exec_*.ntff"))
    return found[-1] if found else None


def view_summary(ntff: str, *, neff: str = NEFF, timeout: int = 600) -> dict:
    """The profile summary for one NEFF/NTFF pair, including `total_exec_time`.

    neuron-explorer refuses a pair whose uuids differ, so a summary coming back at all is also
    proof that this profile belongs to this NEFF.
    """
    p = subprocess.run([NEURON_EXPLORER, "view", "-n", os.path.abspath(neff),
                        "-s", os.path.abspath(ntff), "--output-format", "summary-json"],
                       capture_output=True, text=True, timeout=timeout)
    start, end = p.stdout.find("{"), p.stdout.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"view produced no JSON for {ntff}:\n{p.stdout[-1500:]}"
                           f"{p.stderr[-1500:]}")
    doc = json.loads(p.stdout[start:end + 1])
    if not doc:
        raise RuntimeError(f"view produced an empty summary for {ntff} -- does this NTFF belong "
                           f"to this NEFF?\n{p.stdout[-1500:]}{p.stderr[-1500:]}")
    return next(iter(doc.values()))


def view_parquet(ntff: str, out_dir: str, *, neff: str = NEFF, timeout: int = 1800) -> str:
    """Ingest a profile into parquet tables (`Instruction.parquet` and friends)."""
    p = subprocess.run([NEURON_EXPLORER, "view", "-n", os.path.abspath(neff),
                        "-s", os.path.abspath(ntff), "--output-format=parquet",
                        "--output-file=" + os.path.abspath(out_dir), "--ingest-only"],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"parquet ingest failed:\n{p.stdout[-1500:]}{p.stderr[-1500:]}")
    return out_dir


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
