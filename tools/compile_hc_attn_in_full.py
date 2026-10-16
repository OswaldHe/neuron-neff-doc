"""Compile-only harness for optimization-runs-HCSublayerIn/layers-0-hc_attn_in-full/source.py.

That repo's frozen validator was never written (its inference.py is the stage-2 stub), so this
stands in for it just far enough to get the NEFF: each of 4 ranks slices its hidden columns out
of the recorded tensors at the shapes `submodule/inference.py` pins, traces `kernel`, and runs
it once. No comparison against the golden. Launch with the compile cache pointed at
`build/cache/<sha256(source.py)[:16]>`, the same place the frozen validators use:

    NEURON_RT_NUM_CORES=4 NEURON_CC_FLAGS=--cache_dir=<full>/build/cache/<digest> \
        torchrun --nproc_per_node=4 compile-hc-attn-in-full.py <full>
"""

import os
import sys
from pathlib import Path

import numpy
import torch

FULL = Path(sys.argv[1]).resolve()
WORLD = 4
NTOK, HC, DIM, MIX = 128, 4, 5120, 24
RDIM = DIM // WORLD


def load(stem, dtype, shape):
    return torch.from_numpy(numpy.fromfile(FULL / "tensors" / f"{stem}.bin", dtype=dtype)).reshape(shape)


def main():
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend  # noqa: F401 -- registers the 'xla' backend
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    sys.path.insert(0, str(FULL))
    from source import kernel

    dist.init_process_group("xla", init_method="pjrt://")
    device = torch_xla.device()

    cols = slice(rank * RDIM, (rank + 1) * RDIM)
    x = load("input_0", numpy.uint16, (1, NTOK, HC, DIM)).view(torch.bfloat16)
    fn = load("layers_0_hc_attn_in_fn", numpy.float32, (MIX, HC, DIM))
    host = (
        x[..., cols].contiguous(),
        load("input_1", numpy.float32, (1, NTOK, HC)),
        load("layers_0_hc_attn_in_base", numpy.float32, (MIX,)),
        fn[:, :, cols].reshape(MIX, HC * RDIM).contiguous(),
        load("layers_0_hc_attn_in_scale", numpy.float32, (3,)),
    )
    for t in host:
        print(f"[rank {rank}] {tuple(t.shape)} {t.dtype}", flush=True)
    outs = kernel(*(t.to(device) for t in host))
    xm.mark_step()
    print(f"[rank {rank}] done, outputs {[tuple(o.cpu().shape) for o in outs]}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
