"""Run NEFFs on the device, compare outputs with the golden and with each other bit for bit.

usage: python check.py <workdir> <reference.neff> <candidate.neff> [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("GDN_LOOP_TOOLS", "/home/ubuntu/workspace/gdn-loops/gdn_block_tkg_b1_h128_vh4/tools"))
import device as D  # noqa: E402

work, neffs = sys.argv[1], sys.argv[2:]
m = D.manifest()
gold = D.golden(m)
outs = {}
for n in neffs:
    w = os.path.join(work, os.path.basename(n)[:-5])
    r = D.capture(w, m, neff=n, save_outputs=True, timeout=120)
    if r["returncode"] != 0:
        print(n, "FAILED", r["stderr"][-400:])
        continue
    o = D.read_outputs(w, m)
    outs[n] = o
    errs = {k: float(np.abs(np.asarray(o[k], np.float32) - np.asarray(gold[k], np.float32)).max())
            for k in o}
    print(os.path.basename(n), "max_abs_err", {k: f"{v:.3e}" for k, v in errs.items()},
          "limit", m["max_abs_err_limit"])
ref = neffs[0]
for n in neffs[1:]:
    if n in outs and ref in outs:
        same = all(np.array_equal(np.asarray(outs[ref][k]).view(np.uint8),
                                  np.asarray(outs[n][k]).view(np.uint8)) for k in outs[ref])
        print(os.path.basename(n), "bit-identical to", os.path.basename(ref), same)
