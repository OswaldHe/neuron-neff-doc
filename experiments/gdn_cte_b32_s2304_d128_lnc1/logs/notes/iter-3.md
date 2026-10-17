# iter-3 — the "spill" buffers are DRAM-staged reshapes, and SBUF has a 45 KB/partition hole

## Result: 14381 us, **-0.83% vs the iter-2 NEFF** (14502), **-2.69% vs the iter-0 baseline** (14779)

| NEFF | total_exec_time_us | captures |
|---|---|---|
| iter-2 NEFF = my starting point | **14501.6** | 14498.4 / 14504.0 (med of 3) / 14499.2 / 14909.5 (outlier) |
| + reload reads SBUF directly (phase 1, 1832 DGE) | **14458** | 14454.4 (med 3) and 14463.3 (med 5), 8 captures all in 14386..14476 |
| + DRAM staging moved into free SBUF (320 vars, 2624 DGE) | **14380.7** (med 5) | 14379.6 14380.7 14380.8 14385.4 / 14487.2 |
| + same for all 640 DGE-only vars (3416 DGE) — **shipped** | **14382.1** (median of 11 captures) | 14376.2 14380.8 14380.8 14381.4 14381.6 14382.1 14385.6 14387.1 14387.8 14388.1 / 14472.3 |

`verify.py --runs 3`: PASS, **all runs bit-identical**, `max_abs_err` identical to the baseline to
every printed digit (1.372e-04 / 6.841e-04). Nothing arithmetic changed — only DMA addresses.
`single_core.py`: PASS. Ten of the eleven captures lie in 14376..14388 (one +90 us outlier), and
every one of them is below every capture of the starting NEFF, so the -120 us (-0.82%) is real,
not the 0.5% noise floor.

## The structural finding (this is the useful part)

**The 800 `tmp-buf` DRAM variables are not spills. They are reshapes staged through DRAM.**

* `def.json` has 800 `type: "tmp-buf"` vars, 160,432,128 bytes total = exactly the profile's
  `spill_save_bytes`. 2784 DGE transfers write them (these are the slots iter-1/iter-2 moved
  between hosts — that is what those were all along), 2624 DGE transfers read them back,
  plus 1152 queued `qSPSpillReload0` descriptors read 160 of them.
* **For every one of the 640 DGE-read vars, the save's SBUF source and the reload's SBUF
  destination are different regions** (0 of 640 match). So DRAM is not holding data while SBUF
  is reused — it is a transpose-like **partition/offset redistribution** that a 2-D DGE cannot do
  in one hop, so the compiler does SB -> DRAM -> SB.
* Two shapes, 320 vars each: `1 save + 8 reloads` (save writes 64 partitions x 4096 B
  contiguous, each reload picks one 512-B column out of every partition) and the mirror
  `8 saves + 1 reload` (the reload re-reads 4096 contiguous bytes and casts FP32->BF16).
* `profile.py`'s `spill_reload_bytes` (142 MB) is **not** these reloads (9.4 MB of queued +
  150 MB of DGE); it is the traffic on `qSPSpillReload0`, i.e. mostly the 9664 SB->SB transposes.

**SBUF has a 45,696-byte free run in every partition at [68736, 114432).** Verified exactly:
I enumerated every access pattern of every slot in all 5 streams (compute src+dst, all 8384 DGE,
all queued descriptors), with 0 patterns skipped, into a per-partition byte bitmap.
Used 167,160 of 262,144 bytes; free runs are **[68736,114432) = 45,696 B**, [229368,262144)
(past the usable window — the runtime reserves 0x37FF8), [0,16384) (= `qPoolDynamic`'s
`dynamic_dma.SB_scratch_partition_*`, do not touch) and 128 B at 57216.
Reusable fact for any future iteration that needs scratch SBUF: **11 windows of 4096 B/partition
are free, and at most 3 of the 640 staging live ranges are live at once** (median live range
6.8 us), so the allocation has a 3.6x margin.

## What I shipped, exactly

Two address-only edits to DGE (`PSEUDO_DMA_DIRECT2D`) slots. No slot inserted, deleted, or
reordered; no wait, semaphore, `sem_increment` or dtype changed; `semaphores.py` output
**byte-identical** to the starting NEFF. Final diff vs the iter-2 NEFF: `Activation0.bin`
1308 of 30789 slots, `Pool0.bin` 160 of 7584, `SP0.bin` 3780 of 18756 — 5248 slots, every one a
`PSEUDO_DMA_DIRECT2D` in both versions. `PE0.bin`, `DVE0.bin`, `SP0.json`, `def.json` and every
other member are unchanged.

1. **Phase 1 — 1832 reloads read SBUF instead of DRAM** (`scratch/direct.py`).
   For the 260 `1-save + N-reload` vars where the save's SBUF *source* is provably still intact
   when the reloads run, point the reload's `src` at that SBUF region: build the DRAM->SBUF map
   from the save's two patterns, then transform the reload's base and both `src_step_elem`
   (steps are **bytes**; partition stride 262144, so "16 partitions" = 4194304 — the field is
   32-bit, no overflow). Validity is checked per transfer: each `src_elem_size` run must stay
   contiguous in SBUF and each dim must be affine; skip otherwise. -60 MB of HBM reads.
   *Intactness proof:* for every writer in the program (compute dst, DGE dst, queued `to`) I took
   its (start,end) from the profile and asked whether any of them overlaps the save's SBUF source
   range in partitions **and** bytes **and** time between the save's start and the last reload's
   end. The schedule is static (no branches anywhere in this NEFF) and all addresses are
   literals, so one profile's order *is* the order of every execution: 260 clean, 60 clobbered
   (skipped), 320 excluded for having >1 save.
2. **Phase 2 — all 640 DGE-only staging vars stage in SBUF, not DRAM** (`scratch/window2.py`).
   Give each var a 4096-B window in [68736,114432), chosen by greedy interval colouring of
   [first save start, last access end], and re-point **the saves' `dst`** and **any still-DRAM
   reload's `src`** at it with the same pattern transformation. Same-window pairs are asserted
   disjoint in time (max 3 windows live; 11 available). -242 MB of DRAM traffic in total.
   This needs **no** intactness analysis — it only moves where the staging lives — so it covers
   the 60 clobbered and the 320 multi-save vars that phase 1 could not touch.

## What the measurements mean (read this before optimising DMA again)

* **The win is latency, not bandwidth.** Phase 1 (-60 MB of reads) bought -43 us. Phase 2's first
  half (-151 MB) bought another -77 us. Phase 2's second half removed a further **91 MB of
  now-dead DRAM writes and bought 0 us** (14380.7 -> 14381.4, i.e. nothing). HBM traffic per se
  is not the constraint — confirming iter-1's 27 GB/s observation. What helped is that an
  SB->SB DGE *completes sooner* than SB->DRAM->SB, and those completions
  (`S[8..23]`/`S[40..55]`/`S[56..71]`) sit on the critical path.
* **The 151 MB of saves for the 640 vars is now provably dead** for the 260 phase-1 vars (their
  window is never read) and merely SBUF-local for the rest. Deleting those saves outright is the
  obvious next step, but it is **not** a free win: their completion semaphores are also the WAR
  barrier that lets the next writer overwrite the save's SBUF source. Deleting or shrinking them
  needs that re-pointed, and the +0 us result above says the reward is small.
* **Do not convert the queued (`SP0.json`) reloads to SBUF sources.** I did it (153 of 160 vars,
  1096 descriptors, 4-dim patterns, bit-exact) and it **regressed +199 us** (14454 -> 14653, DMA
  busy +0.5% instead of -4%). Those descriptors move 32-byte runs; reading them from SBUF with a
  partition-crossing step is worse for the DMA engine than one linear DRAM read. Reverted.

## The next step I got 90% of the way through: fuse the save and the reload

For the 320 `N-saves + 1-reload` vars the reload is a pure window->destination copy with an
FP32->BF16 cast, so **each save can do that cast itself and write the reload's SBUF destination
directly**, removing a whole DMA hop (and its completion) from the chain. I built it
(`scratch/fuse.py`): map window byte -> destination byte at *element* granularity from the
reload's two patterns (src esz 4096 = 1024 FP32 elements -> dst esz 2048 = 1024 BF16, element i
to element i in iteration order), then rewrite each save's `dst` base, both `dst_step_elem`,
`dst_elem_size` (512 -> 256) and `out_dtype` (FP32 -> BFLOAT16). 319 of 320 groups pass every
check (coverage of every destination byte; and the hazard check that nothing reads *or* writes
the destination between the first save's start and the reload's end — exactly one access may
intersect, the reload's own write; 1 group fails that and is skipped).

**The rewritten saves are valid and run.** I shipped them alone as a diagnostic (reloads left
copying the now-stale window): the device **ran it in 10.4 s with finite outputs** and of course
the wrong values (`max_abs_err` 2.8e-02 / 2.6e-01). So a DGE that casts FP32->BF16 with
`src_elem_size` 512 / `dst_elem_size` 256 over a partition-strided SBUF destination is accepted.

**What blocks it is retiring the reload.** iter-1's suggested trick — "shrink the transfer to a
dummy so it still signals" — **does not work for a DGE.** Three variants, each applied *alone*
(no save edits), each **hangs** the model:

| dummy variant | result |
|---|---|
| `src/dst_num_elem = [1,1]`, `step = [0,0]`, `esz` = one element, src=dst=its own window | `status 1004`, model stop timeout (30 s) |
| `num_elem = [1,1]`, `esz` 256/128, dst = the free 128-byte run at offset 57216 | hang |
| **shape preserved** (same `num_elem [64,1]`, same `step_elem [262144,1]`), `esz` 4096/2048 -> 64/32, dst = a window reserved as trash | **`status 1203`**, hang — and this one was applied with no other change at all |

So shrinking a DGE's `elem_size` stops it signalling (or stops it completing), while changing its
*address*, *steps* and *dtypes* is fine (both of this iteration's shipped passes do exactly that,
and the saves-only diagnostic above does too). Guess at the mechanism: `sem_increment` is 16 on
every DGE and the queue has `num_queues: 16`, so the transfer is probably split over 16 rings
each contributing 1, and a transfer with too few bytes per ring never reaches 16. Not tested.

**Therefore the route to finish the fusion is to delete the reload slot and pay its +16 some
other way**, not to shrink it. Check whether `EVENT_SEMAPHORE`'s
`events_extended_sem_update_value` can add 16 in one slot (it is a value field, unlike the plain
`events_update_mode` which adds 1); if it can, replace each reload with one `EVENT_SEMAPHORE`
that keeps the reload's wait and adds 16 to the reload's `semaphore`. 319 instances, all the
analysis above already written. Expected gain: one DGE hop plus a 128 KB SB->SB transfer removed
from 319 places on the dependency chain.

## Hypotheses REFUTED this iteration (with the evidence; do not repeat)

1. **`def.json` `dma_queue.qPoolDynamic.dynamic_dma` cannot be removed** (the iter-2 review's
   top recommendation, to turn `qPoolDynamic` from the software-dynamic DMA path to hardware).
   Deleting the 3-line block so the entry matches `qActDynamicHW` -> **`NRT_INVALID in
   nrt_load_util`**. The block is present on `qPoolDynamic` in every NEFF of the doc corpus and
   on no other queue; it is mandatory. The queue table is validated at load (iter-2 already lost
   a 5th queue, a new owner and an owner rename to this). **The `def.json` DMA-queue axis is now
   fully closed.**
2. **`dma_configs_priority_class` (3 bits, 0 on all 8384 DGE slots) has no headroom.**
   Setting it to **7** on every DGE slot -> the model **hangs** (`status 1004`, model stop
   timeout), so the field does reach the hardware. Setting it to **1** on every DGE slot runs,
   is bit-exact, and measures **14512 us vs 14504** — i.e. nothing, or a touch worse. Since the
   8285 queued transfers carry no priority field at all (so they sit at the default), DGE was
   already differentiated from them at 0; if lower means higher priority, 0 is already the best
   value available. Nothing to gain here.
3. **Redundant `LDWEIGHTS` do not exist.** 14656 `LDWEIGHTS` feed 17184 `MATMUL`s (so the
   compiler already reuses the stationary operand 2528 times), and only **31** `LDWEIGHTS`
   repeat the previous one's full key (address+pattern+rows/cols+groups+dtype+modes) with only
   `MATMUL`s in between. 31 x ~100 ns = 3 us. Dead end.
4. **No activation-table churn.** The Activation stream has exactly **one**
   `ACTIVATION_TABLE_LOAD` (slot 1) in 30789 slots, so there is no table-reload penalty to
   remove.
5. **There is no dead work in the DRAM staging.** My first read of the data said 640 of the 800
   `tmp-buf` vars were written and never read (151 MB of dead stores). That was wrong — my
   source/destination histogram was truncated at the top 20 entries. Every one of the 800 is
   read (2624 DGE + 1152 queued descriptors). Check read/write sets exhaustively, not by
   `most_common()`.

## The transpose lever (hints.md): the hardware feature exists and is already in use

hints.md proposed deleting the 8125 `qSPSpillReload0` SB->SB DMA transposes by swapping a
matmul's stationary and moving operands. Two findings that change that plan:

* **PE can already load the stationary operand transposed, and this NEFF does it**: `LDWEIGHTS`
  carries `transpose_mode`, `TRANSPOSE_MODE_ENABLED = 1` exists for v3, and **1152 of the 14656
  `LDWEIGHTS` in this NEFF are already `ENABLED`**. So no operand-role gymnastics are needed —
  for a transpose that only feeds an `LDWEIGHTS`, the substitution is a one-field edit plus
  re-pointing the address at the untransposed tile. **3960 `LDWEIGHTS` wait on `S[24..39]`**,
  i.e. consume a queued transpose: 2376 with rows=128/num=64, 1134 with 64/64, 450 with
  rows=128/num=128 and a `[-1,0,-32]/[32,1,4]` pattern.
* **The blocker is the reversal, not the orientation.** Every one of those 3960 reads its tile
  with `src_mem_pattern_step_elem = -1` (free axis read backwards). For transpose dma[0]:
  A lives at SB part 0..63 off 227072 (64p x 128 bf16), the DMA writes T[p][i] = A[i][p] at
  off 16896 (128p x 64), and `LDWEIGHTS` slot 7 reads T backwards, so the array holds
  `stationary[p][i] = A[63-i][p]`. `transpose_mode` on A gives `stationary[p][i] = A[i][p]` —
  the array's column index comes from A's **partition** axis, and an access pattern cannot
  reverse partitions. Reversing it instead in A's producer means reversing *that* matmul's
  stationary columns, i.e. a chain edit (upstream `LDWEIGHTS` step sign -> A's partition order ->
  this `LDWEIGHTS`), and SB 16896 is a scratch tile with 2090 writers and 1782 readers so each
  instance has to be paired temporally, not statically. Worth an iteration; budget it whole.

## Where the time is now (shipped NEFF, span 14381 us)

Engine busy: Activation 8265 (57.5%), DVE 5679 (39.5%), DMA 5390 (37.5%), PE 3190 (22.2%),
SP 2758, Pool 2751. Unchanged in shape from iter-2: no engine is near 100%, so this is still a
cross-engine dependency chain. The iter-2 review's critical-path decomposition still applies,
minus roughly the 120 us this iteration took out of the DGE-completion terms.
The two big remaining blocks are **~6.7 ms of on-path instruction execution** and the
**~4.7 ms of queued SB->SB transposes** (of which ~1.3 ms is the hardware doorbell->data floor,
median 725 ns, min 693 ns over 8285 transfers).

## What I would do next, in order

0. **Finish the save/reload fusion** (section above): 319 instances are fully analysed and the
   rewritten saves are proven to run on the device; all that is missing is a legal way to retire
   the reload slot (delete it and add 16 to its semaphore from an `EVENT_SEMAPHORE`, since
   shrinking it hangs). This is the cheapest remaining win and the groundwork is done.
1. **The `transpose_mode` substitution.** It is the only remaining multi-ms lever and the
   hardware support is now confirmed present and in use. Start by pairing one transpose with its
   `LDWEIGHTS` consumer through `S[24..39]` wait values (static: wait value `16*(id//16+1)` on
   `S[24+id%16]` names the transfer id exactly — no timeline needed), then test whether the
   reversal can be pushed into the producer.
2. **Delete the 151 MB of now-SBUF-local staging saves** for the 260 phase-1 vars whose window is
   never read. Needs each save's completion semaphore waiters re-pointed (they are the WAR
   barrier for the save's SBUF source). Expect little: the same change in traffic bought 0 us.
3. **Do not** spend budget on: DGE host/queue ratios (iter-2, at the optimum), `def.json` DMA
   queues (refutation 1 + iter-2 refutation 6, fully closed), DMA priority (refutation 2),
   engine occupancy (iter-1 refutation 1), trigger/wait tightness (iter-1 refutation 2),
   DGE coalescing or `dge_op = DMA_TRANSPOSE` (iter-1 refutations 3 and 7), queued-path SBUF
   sources (measured +199 us above).

## Tooling notes

* `scratch/` scripts in this worktree are not preserved; the recipes above are complete.
  `dge2.py` (decode all DGE with `PSEUDO_ADDR8` resolved: marker byte 0x10=VAR, 0x20=TBL, and the
  `DGETable0` list in `def.json` maps a table index to a var), `sbuf.py` (every compute slot's
  SBUF/PSUM src+dst), `reshape.py` (intactness check), `direct.py`, `window2.py`.
* `isa.Instruction.set` takes a **whole list** for array fields (`set('src_step_elem', [a,b])`);
  `('src_step_elem', 0)` raises `KeyError`. `ins.name` is the mnemonic, `ins.opcode` the number,
  `ins.as_dict()` everything.
* `neff_diff.py` takes minutes on a changed `SP0.json` (it unified-diffs the pretty-printed JSON
  of 8285 transfers). Diff the `.bin`s, not that member.
* Do not leave a `def.json.orig` inside `work/payload/sg00/` — `neff.py pack` will package it as
  a 35th member.
* Capture noise on an identical binary: 4 of 5 captures within 1 us, with one +90..100 us
  outlier per 5 (seen on every candidate this iteration). Take the median of 5 and look at the
  cluster, not the spread.
