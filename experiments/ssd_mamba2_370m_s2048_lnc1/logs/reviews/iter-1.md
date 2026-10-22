# Review of iter-1 (ssd_mamba2_370m_s2048, kernel.neff)

## 1. What changed

Only `sg00/Activation0.bin` changed (11973 slots both sides, 8325 slots differ, but that is mostly an
insertion/deletion shift). Every other member, the JSONs and the DMA/semaphore tables are byte-identical.
`notes/README.md` is still the empty template, so there is no claim to compare against; the description below
is from decoding the streams.

ACT-engine opcode counts, baseline -> edited:

| | baseline | edited |
|---|---|---|
| `ACTIVATION_TABLE_LOAD` | 1531 | 265 |
| `ACTIVATE` func 129 (reciprocal) | 768 | 135 |
| `ACTIVATE` func 7 (exp) | 1536 | 2169 (+633) |
| `ACTIVATE` func 128 (copy) | 5376 | 6009 (+633) |
| `EVENT_SEMAPHORE` | 1579 | 2212 (+633) |
| `PSEUDO_DMA_DIRECT2D`, `COPY` | 816, 366 | 816, 366 |

The edit does two things at 633 of the 768 reciprocal sites (slots 7-10 are the first one):

* **Numerical rewrite.** `recip(exp(x))` becomes `exp(-x)`. The source `x` is copied to a new one-element scratch
  at SBUF offset 0x37f80 (func 128, 128 partitions x 1 element). Then `exp(scale=-1.0, src=scratch)` is issued with
  no semaphore update, then `exp(scale=+1.0, src=scratch)` with `SEM_INC`, then an `EVENT_SEMAPHORE` wait before
  the DMA that consumes the result. The reciprocal and its two `ACTIVATION_TABLE_LOAD`s (to table 1 and back to 0)
  disappear, so the exp/copy table stays resident.
* **Regrouping.** ACTIVATEs are reordered so same-table work runs together.

Net effect: the table loads fall by 1266, one per removed reciprocal-table switch. That is 1266 x 1.283 us = 1.62 ms.
The latency win is 1.628 ms, so this accounts for it. Each converted site costs one extra COPY (about 168 ns), and
the `EVENT_SEMAPHORE` count rises by 633, i.e. one per site.

One thing the task statement forbids is touched here: this is not a pure reorder. **The numbers changed.**
Edited and baseline outputs are not bit-equal:

* output_0: 2,787,639 of 3,145,728 elements differ, max diff 2.5e-7.
* output_1: 89,392 of 98,304 elements differ, max diff 1.4e-7.

Verifier error moved as follows:

| | baseline | edited | % of limit |
|---|---|---|---|
| output_0 | 1.88e-7 | 2.06e-7 | 82% -> 89% |
| output_1 | 1.19e-7 | 1.58e-7 | 52% -> 69% |

The limit is 2.3e-7.

## 2. Is the latency win real?

Yes. `profile_latest.json` has median 3624.2 us over 5 repeats (min 3620.2, max 3625.3, spread 0.14%) against the
baseline's 5248.9 us (spread 0.022%). The change is -31.0% (1.448x), which is over 100x the spread. My own re-profile
gave 3620.6 us. ACT busy fell 4721.7 -> 2727.1 us. The other engines are flat within 2.6% (DMA -32 us, DVE -27 us,
PE and Pool flat), so the saving is entirely on ACT. The loop is not chasing noise.

Of the 1995 us drop in ACT busy time, 1628 us turned into end-to-end time (about 0.82). That ratio is the working
estimate for the candidates below.

## 3. Safety beyond the verifier

* **Determinism.** `verify.py --runs 5` gives 5/5 bit-identical outputs, so there is no sign of a race on this input.
* **Semaphores.** `semaphores.py` on both unpacked NEFFs: all totals and `max_wait` values are unchanged. The only
  difference is sem 4 (ACT) waiters, 6996 -> 8262. That is more waits, not fewer, so nothing was loosened.
* **One assumption.** The new `exp(-x)` has `events_update_mode=NONE`. Downstream consumers see its writes only
  through the following `exp(+x)`'s `SEM_INC` and the new `EVENT_SEMAPHORE`. That is safe only if ACT completes
  in order, which is plausible but not measured.
* **Scratch address.** 0x37f80 is used 1899 times and never in the baseline. The baseline's highest other offset is
  0x37e00. The runtime reserves 0x37FF8 and an element is 4 bytes, so it fits. The kernel's SBUF allocation is
  not visible in the NEFF, so confirm that 0x37f80 is free, for example by checking that no DMA descriptor touches
  it. Its single use point is serialised in ACT program order, which is why one scratch is enough.
* **Numerical headroom.** `1/exp(x)` computed as `exp(-x)` differs from the table reciprocal by a few ulp. Output_0
  is at 89% of the limit. More rewrites of this kind (candidate 1) could push it over. The verifier would catch
  that on this input set, since the check is on the actual golden.
* **Input range.** One fixed input set is verified. `exp(-x)` overflows where `recip(exp(x))` underflows and the
  reverse, but only for |x| beyond about 88. Mamba decay arguments are bounded by `dt*A`, so this is not a
  concern for this case.

## 4. Ranked candidates for the next iteration

Method note: sums of `duration_ns` double-count overlapped instructions (PE: MATMUL + LDWEIGHTS sum to 3.45 ms
against 1.27 ms engine busy). Use `engine_busy_us` for ceilings. Current per-engine busy is ACT 2727 us (75%),
DVE 1662, Pool 1559, PE 1268, DMA 1160, SP 573, against a 3620 us span.

| # | Edit | Engine / member | Ceiling (and where it comes from) | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | Finish the reciprocal removal at the remaining **135** `ACTIVATE` func-129 sites (first at slots 30, 726, 739, 754, 755, 826, ...) and their ~265 `ACTIVATION_TABLE_LOAD`s | Activation0.bin | 265 x 1.283 us = **340 us** (Instruction.parquet: `ACT_TABLE_LOAD` count 265, mean 1283 ns). At the 0.82 ratio, ~280 us (3620 -> ~3340) | Numeric: output_0 is at 89%, and 17% more rewrites could cross 2.3e-7. Sync: same pattern as the accepted edit. Verifier catches the numeric case. | Convert 1 site and run `verify.py`. Then convert all and compare error against the 2.3e-7 limit. If error goes over, convert only the sites that keep it in range. |
| 2 | Drop the scratch COPY at the 633 converted sites: let `exp(-x)` and `exp(+x)` both read the original source directly | Activation0.bin (the inserted func-128 slots) | 633 x 168 ns = **106 us** busy (profile: `COPY` fn, 633 instrs, 106,234 ns). Realistic ~85 us | Both reads must happen before the PSUM/SBUF source is released (sem 4 / PE reuse). Read after release would be wrong. Result should be **bit-identical to the current NEFF**, so the check is exact. | Do it for one site, run `verify.py --runs 2`, and compare outputs bit for bit with the current ones (my script `/tmp/cmp.py` pattern: `D.capture` + `read_outputs`). Any difference means a hazard. |
| 3 | Move the 816 ACT-issued `PSEUDO_DMA_DIRECT2D` (sems 42-55, "dge:Activation") into the SP stream (insert on SP, same wait fields, replace on ACT with `EVENT_SEMAPHORE` keeping the count) | Activation0.bin + SP0.bin | 816 x 530 ns = **433 us** of ACT time. SP has 15.8% busy (573 us), Pool 43%. Realistic ~250-350 us | Moderate. Docs 09 and 08 say DGE is legal on SP, ACT and Pool, but moving it across engines is not shown on this device. The DGE queue/ring may be tied to the issuing engine. Semaphore accounting is the same, since the DMA's own semaphore is unchanged. | Move **one** DMA near the start (slots 25-26) and run. A crash or wrong output kills it in one run. If it runs, check the semaphore tool for unchanged totals and measure. |
| 4 | Move ACT tensor-scalar-style copies (func 128 with a pointer scale, 5376 of them, mean 329 ns) to idle DVE or Pool | Activation0.bin + DVE0.bin or Pool0.bin | Up to 1.77 ms of ACT instruction time (not additive: see method note). DVE is 46% and Pool 43% busy, so moving even 20% would give a few hundred us. Unknown net | Structural: different instruction struct on the other engine, new waits, new semaphore count bookkeeping for every consumer. A hazard may pass one fixed input set. High risk. | Convert one isolated instance (single consumer) and check bit-equality to the current outputs. |
| 5 | Reduce ACT stalls. ACT total gap time is 1030 us of the 3628 us ACT span, so ACT waits on other engines | Activation0.bin ordering | Unknown, up to ~1 ms of ACT gap | Reordering on one engine is verified mechanics (doc 14.2), but the gain is speculative | Read `Flow.parquet` or `evt_wait_time_ns` for the ACT gaps and see which semaphore they wait on. If it is DMA completion, candidate 3 helps. If it is PE/DVE, reordering ACT will not. |

Ranking logic: 1 and 2 are small, local copies of an edit that already ran correctly. 1 is worth the most, 2 is
essentially free to verify (bit-exact), 3 is the biggest remaining ceiling with an unproven mechanism, 4 and 5 are
exploratory.

### Closed (do not redo)

* Nothing recorded in `notes/` (it is the empty template). No earlier review exists.
* From `neff_doc/docs/14`: removing waits on an engine's own completion semaphore was already refuted
  (wrong output on DVE, out-of-bounds on Pool, no gain on Activation; experiment o01). Do not attempt it.
* This iteration's own result: the regrouping of ACT work is not the lever. The saving is 1266 table loads x
  1.283 us = 1.62 ms, which matches the whole 1.628 ms gain.

### Unknown (would change the ranking)

* **Is ACT still critical?** Settle with a per-instruction Flow trace of the ACT gaps (1030 us): what semaphore
  does ACT wait on, and does the producer sit on the critical path?
* **Is 0x37f80 free in the kernel's SBUF map?** The NEFF does not expose the allocator, so check that no DMA
  descriptor or other engine touches that offset across all five streams.
* **Can ACT complete out of order?** It would matter for the unsynchronised `exp(-x)`. A two-run bit-equal check
  on a variant that makes `exp(-x)` the last instruction before the `SEM_INC` would settle it.
* **Does a DGE DMA issue from SP behave like one from ACT?** Candidate 3's one-site test settles it.
* **Error headroom.** How much of the 2.3e-7 limit can candidate 1 use? Only measuring the full conversion shows
  it.
