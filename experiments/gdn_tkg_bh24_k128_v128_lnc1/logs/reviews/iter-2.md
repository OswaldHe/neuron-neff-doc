# Review: iter-2 (gdn_tkg_bh24_k128_v128, LNC=1)

## Baseline for the comparison

The working `kernel.neff` is not the committed one. HEAD (iter-1) is 79.6 us and the working tree is a further edit on top of it. I diffed against both. `notes/` is empty (only the README template), so there are no notes to check the change against.

## 1. What changed

Diff of working tree against HEAD (iter-1). Five members changed: `Activation0.bin`, `DVE0.bin`, `PE0.bin`, `Pool0.bin` (8 slots differ, so some descriptors were retargeted) and `SP0.bin`. Slot counts:

| Engine | Slots, HEAD → now | Opcode change |
|---|---|---|
| ACT | 191 → 170 | `CAST` 72 → 36; `EVENT_SEMAPHORE` 7 → 22 |
| DVE | 159 → 166 | 7 new `TENSOR_TENSOR_ARITH_OP` (dst 210688 + 768·k, k = 0..6) |
| PE | 152 → 163 | `EVENT_SEMAPHORE` 3 → 14 |

- **Wide casts.** The per-head narrow `CAST S→S_bf` ([128,1] elements, ~298 ns each) is replaced by 4 wide casts of [128,6] (~826 ns each). Their dst addresses are 147456, 148992, 150528 and 152064. ACT cast cost drops from about 24×0.3 us to 4×0.83 us.
- **Waits made explicit.** Inline waits on the ACT `CAST` slots moved into standalone `EVENT_SEMAPHORE` slots. Slot 14, for example, has `events_wait_mode` WAIT → NONE, and there is an `S[6]>=10` wait slot immediately before the wide cast at pc 77. PE gained standalone waits as well.
- **Consistency with the claim.** I can't say whether this matches what the agent meant, because no notes were written. The commit message for iter-1 describes iter-1, not this change.

Arithmetic is unchanged:

| Output | `max_abs_err` | Share of limit |
|---|---|---|
| `output_0` | 2.322e-02 | 83% |
| `output_1` | 1.633e-03 | 6% |

These are the same digits as baseline, so the edit is a pure schedule/placement change.

## 2. Is the latency change real?

| Run | Median (us) | Spread | Samples (us) |
|---|---|---|---|
| iter-2 `profile_latest.json` | 74.615 | 1.30% | 74.43, 74.48, 74.62, 74.64, 75.40 |
| My re-measure | 74.388 | 1.49% | min 74.28, max 75.38 |
| Baseline | 153.46 | 0.79% | |
| HEAD (iter-1, from commit message) | 79.64 | n/a | |

- **Against baseline.** −51.4%, about 2.06×, and 40× the spread. This is real.
- **Against HEAD.** −5.0 us (−6.4%), about 4× the ~1.3% spread. Real, though I did not re-profile the HEAD NEFF here.
- **Noise.** No sign the loop is chasing noise at this step size.

## 3. Safety beyond the verifier

- **Semaphore supply.** `semaphores.py` on the unpacked NEFF shows every semaphore's `max_wait` equal to `total_inc`: s4 120/120, s5 74/74, s6 100/100, and the DMA semaphores 16/16 and 128/128. No wait exceeds its supply, so no hang. Supply on s4 dropped 140 → 120 (the removed casts) and the waits were renumbered consistently.
- **Waits made stricter, not looser.** Inline waits were split into standalone `EVENT_SEMAPHORE` slots. I found nothing removed.
- **Not covered.**
  - `semaphores.py` checks supply only. It does not check ordering, WAR hazards on the reused S_bf tiles, or whether the `s4>=v` prefix semantics still cover everything the reordered casts used to cover.
  - `verify_latest.json` shows `runs: 1`. One passing device run can hide a race. The previous review corpus (`neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2`) records that reordering ACT silently dropped implicit coverage and produced garbage, so a cycle/WAR check is worth running.
  - `verify.py --runs 5` should give bit-identical outputs.
- **Error margin.** `output_0` sits at 83% of the 0.028 limit. That is unchanged from baseline, but any candidate that alters rounding has almost no room. Keep every candidate bit-exact.

## 4. Where the time is now

I re-profiled with `--parquet`. The trace spans about 87 us; about 12 us of that is runtime prologue and epilogue, which leaves about 74 us as the metric.

**Engine busy time.**

| Engine | Busy (us) | Share of 74.4 us |
|---|---|---|
| ACT | 55.2 | 74% |
| DVE | 40.5 | 54% |
| PE | 18.8 | 25% |
| Pool | 14.7 | 20% |

ACT is the critical engine.

**Per-bucket busy, from the instruction table.** In the 30–80 us body ACT is 75–91% busy per 5 us bucket.

**ACT work by cost.**

| Work | Count | Cost | Total |
|---|---|---|---|
| Scale-pointer `ACTIVATE COPY` | 24 | 490 ns each | 11.8 us |
| `CAST` (narrow S→S_bf and others) | 32 | 298 ns each | 9.4 us |
| `ACT_TABLE_LOAD` | 6 | 1.283 us each | 7.7 us |
| Per-head rstd chain: `FUNC_30` square-accumulate, `ACTIVATION_READ_ACCUMULATOR`, `FUNC_29` | 24 per head, per op | 266 + 293 + 202 ns | 18.3 us |

The three ops of the rstd chain are listed together because they run as one chain per head.

**Head (10.5 → ~29 us).** This phase is a chain of tiny [24-element] gate ACTIVATEs (~0.2 us each) separated by 5 table loads (16.1, 17.8, 19.3, 22.2 and 29.3 us, ~6.4 us). It then waits on one 3.36 us DVE op, the wide `S *= exp(g)` multiply (pc 83, 22.9 → 26.3 us). ACT idles 22.85 → 26.3 us waiting on it. The first matmul starts at 28.0 us.

**Tail.** The last compute ends at 84.07 us. The last state-out DMA is issued at 82.35 us and everything completes at about 85.9 us, so the tail after compute is about 1.8 us.

## Ranked candidates for the next iteration

All candidates must stay bit-exact. After any reorder, run a cycle check and a happens-before check, then `verify.py --runs 5` with identical outputs required.

### C1: Move the 24 scale-pointer `ACTIVATE COPY` ops from ACT to DVE (best)

- **Where.** `Activation0.bin`, the `ACTIVATE` slots with `activation_func=COPY`, `scale_src=POINTER` and dst bf16 0x26600+0x100·h. These are the ~490 ns ones at pc 81, 119, 122, 133, 135, 144, 147 and so on; there are 24.
- **What changes.** Delete them from ACT. Insert 24 `TENSOR_TENSOR_ARITH_OP MULTIPLY` slots in DVE (clone an existing mixed-dtype DVE slot such as DVE pc 137). src0 is the fp32 diff, src1 is the fp32 scale with `step_elem=0` and 128 elements (broadcast), `out_dtype` is bf16. Fix the DVE s6 numbering, the ACT s4 numbering, and the PE waits on the produced tile.
- **Mechanism.** This is established on the device by the same-kernel experiment in `neff_doc/experiments/gdn_tkg_bh24_k128_v128_lnc2/logs/notes/iter-2.md`. An ACT `COPY` with a scale pointer is bit-identical to a DVE fp32 multiply, and a scale-pointer ACTIVATE costs ~490 ns versus ~290 ns for a cast. Zero-stride broadcast is in `docs/05-memory-and-addresses.md:75`. That experiment took 55.6 → 50.7 us (−9%) from the same lever.
- **Expected gain / ceiling.** Expect 3–6 us. The ceiling is 11.8 us of ACT busy time (24×490 ns, from the profile). Actual gain is capped by the head-to-head ACT→DVE→PE latency, and by DVE rising from 40 to about 47 us busy.
- **Risk.** Medium. The `s4>=v` prefix trap applies: when an ACT slot is deleted or reordered, waits that relied on program order lose that coverage. A wrong order can also hang the device. The verifier catches a gross error but not a rare race.
- **Cheapest falsifier.** Convert one head (one ACT slot → one DVE slot), run `verify`, and confirm the output is bit-identical. Then compare ACT idle in `Instruction.parquet`. If `max_abs_err` moves a digit, the candidate is dead.

### C2: Batch the per-head rstd step (`FUNC_29`) across 6 heads

- **Where.** The ACT `ACTIVATE FUNC_29` slots (202 ns each, scale 1/128, bias eps), plus the `ACTIVATION_READ_ACCUMULATOR` dst addresses.
- **What changes.**
  1. Point the `READ_ACCUMULATOR` destinations of 6 consecutive heads at 6 contiguous fp32 words instead of the rotating 0x4000, 0x4008 and 0x5dd8.
  2. Replace the 6 `FUNC_29` ACTIVATEs with one 6-element ACTIVATE.
  3. Redirect the consumer DVE `SCALAR_TENSOR_TENSOR` pointer reads to the new addresses.
- **Mechanism.** The op is purely elementwise, so a [6] vector gives bit-identical results. `FUNC_29` shares a table with the rest of the body, so there is no new table load.
- **Expected gain / ceiling.** About 3.6 us of ACT busy time (24 → 4 ops, 18×202 ns). The ceiling is the 4.8 us total of the 24 `FUNC_29` ops, minus added group latency (the head must wait for the group's last sum). Expect 1–3 us. Unknown beyond that.
- **Risk.** Medium. This changes DVE operand addresses and the s4/s6 wait counts. The batched value must not be read before the last `READ_ACCUMULATOR` of the group. A wrong wait could pass the fixed input and still be a race.
- **Cheapest falsifier.** Do one group of 2 heads. Check bit-identical output and measure the ACT time saved.

### C3: Pipeline the 3.36 us wide DVE `S *= exp(g)` multiply

- **Where.** DVE pc 83, `TENSOR_TENSOR` src0 0x1c000 [128 channels × 24 × 128 elements] with a broadcast src1. Also the consumer wide casts in ACT at pc 77.
- **What changes.** Split it into 2–4 DVE ops (24 heads → 12/12, or 6/6/6/6). Each chunk is only a start-address and `num_elem` change. Let each wide cast and the PE matmuls of its heads start as soon as its chunk is done.
- **Mechanism.** The same pattern is the "pipelined in two 6-head halves" step in the `lnc2` experiment (iter-4), where it was neutral alone but was a prerequisite for the later step. Here ACT idles 3.5 us (22.85 → 26.3) and PE starts at 28.0.
- **Expected gain / ceiling.** Up to ~2.5 us (3.36 us minus a ~0.85 us first chunk). Expect 1–2 us.
- **Risk.** Medium–low. Added DVE slots shift every later s6 wait count by the number of slots added. Missing one hangs the device or reads stale state.
- **Cheapest falsifier.** Split into 2 only. If the first PE matmul doesn't start earlier in the trace, drop it.

### C4: Merge or relocate the remaining narrow `CAST` ops (do after C1)

- **Where.** About 28–32 ACT `CAST` slots at ~298 ns with 128 elements; the 4 wide ones are already merged.
- **What changes.** Either merge into wider casts per consumer-deadline group, or move them verbatim to DVE (`CAST` is a valid DVE opcode, per `lnc2` iter-2). Do not move them to Pool.
- **Mechanism.** `lnc2` iter-6 split the casts by deadline: −1.4 us. A 6-head wide cast is 826 ns versus 6×298 = 1.8 us.
- **Expected gain / ceiling.** ≤ 8–9 us ACT busy (28–32 × 298 ns); practical 1–3 us. DVE headroom is limited once C1 is applied (about 47 us busy of 74).
- **Risk.** Medium. PE waits on these producers, so the waits need tightening.
- **Cheapest falsifier.** Read DVE busy after C1. If DVE exceeds ~60 us, don't move more work onto it.

### C5: Finer last state-out DMA group

- **Where.** `Pool0.bin`: 8 `PSEUDO_DMA_DIRECT2D` state-out slots; the last (Pool pc 69) issues at 82.35 us. Also the matching wait in SP/Pool and the DVE s6 count.
- **What changes.** Split the last group, so the final transfer is 1 head (~65 KB) rather than ~3.
- **Mechanism.** The `lnc2` iter-4 notes show state-out DMA splitting gave −1.65 us when the tail was 2.6 us. Here the tail is ~1.8 us after compute and the split is already 8-way.
- **Expected gain / ceiling.** Ceiling ≈ 1.8 us (last compute end 84.07 us to completion ~85.9 us). Expect ≤ 1 us.
- **Risk.** Low–medium. The DMA semaphore total (128) changes, and so does the final wait value.
- **Cheapest falsifier.** Check the DMA packet timeline for the last transfer: if it is already mostly overlapped, stop.

### C6 (low): Avoid some of the 5 head `ACT_TABLE_LOAD`s

- **Where.** Reordering the tiny gate ACTIVATEs (pc 67–81) so same-table functions are adjacent.
- **Why low.** The `lnc2` experiment bottomed out at 6–7 loads per core; this NEFF has 6. Each load removed is at most 1.28 us.
- **Falsifier.** Check the table-set JSON for which functions share a set. If no two of the gate functions share a set, there is nothing to gain.

## Closed

These are findings from the `lnc2` experiment notes. They are about a different kernel, so treat them as strong priors, not proof.

- **Casts to Pool.** Moving the wide head casts to Pool cost +2.8 us (`lnc2` iter-6 A) and +0.2 us for one cast (B). Closed.
- **Folding delta×beta into the ACT cast.** Neutral (38.66 vs 38.59 us, iter-5 A). Closed.
- **Deferring DVE ops past the next updaters** (iter-5 D/E): 38.53 and 38.74/39.50 us. Closed.
- **Pool for DVE `out*silu` sinks.** The verify failed (iter-6 F). Closed.
- **Folding the `out` sinks.** +0.37 to +0.55 us (iter-7 A/B). Closed.
- **Earlier `Q_p` DGE order.** +0.45 us (iter-5 B). Closed.

## Unknown (measure to settle)

- **What actually gates the head.** Compute it with a critical-path pass over `Instruction.parquet`: nodes = (engine, slot), edges = program order plus every semaphore producer→waiter. It decides whether C3 or C1 gives the larger share, since the head phase is ~18 us before the body.
- **DVE headroom.** After C1, DVE busy and stalls decide how much of C4 fits. Measure it with a single-head trial.
- **Whether iter-2's edit is race-free beyond one run.** Run `verify.py --runs 5` on the current NEFF, plus a WAR check on the reused S_bf tiles.
- **Table-set membership.** Which activation functions share each `act_tbl_sel` set — from the table JSONs — settles C6.
