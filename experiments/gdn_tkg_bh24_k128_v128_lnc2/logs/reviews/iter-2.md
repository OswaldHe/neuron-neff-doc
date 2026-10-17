# Review of iter-2 (kernel.neff, working tree vs HEAD df60817 and vs baseline/)

Everything below was read from the NEFF with `neff_diff.py`, `isa.py`, `semaphores.py`
(unpacked at /tmp/u) and a fresh `profile.py --repeats 3 --parquet` run. `notes/` is empty
(only the template README), so the "what the agent claims" side is the commit log plus the
task text. Nothing was written to notes this iteration. That is itself a defect: the next
agent inherits no record.

## 1. What changed

Against HEAD (the 55.6 us commit, which already hoisted the silu activations), 10 members
changed. Each is the same five on **both** `sg00` and `sg01`:

| member | slots HEAD -> now | what |
|---|---|---|
| `Activation0.bin` | 140 -> 93 | 47 fewer ACT instructions: per-head work is re-batched or removed. Scalar instruction count 385 (baseline) -> 238. The wide `ACTIVATE` func 36 (silu, 1536 elem) plus a wide `CAST` 1536 FP32->BF16 remain. The head now runs `ACT_TABLE_LOAD` sel 4,1,2,0,3,2 (6 loads, unchanged from HEAD). |
| `DVE0.bin` | 95 -> 110 | 15 more DVE instructions, so some ACT work moved to DVE. DVE is now 75% busy. |
| `PE0.bin` | 80 -> 80 | 51 slots differ. Same count, so addresses and wait/update fields were re-pointed. |
| `Pool0.bin` slot 4 | 1 slot | `events_semaphore_value` 49 -> 63 |
| `SP0.bin` slot 9 | 1 slot | `events_semaphore_value` 52 -> 66 |

The Pool and SP edits are the right ones. Both wait on DVE's semaphore 6, whose total
increments went 52 -> 66 (DVE gained 14 incrementing instructions), so the end-of-kernel waits
were re-based correctly. ACT sem 4 total is now 63 (was 100).

**Both-subgraph check: passed.** I decoded sg00 and sg01 stream by stream. For all five
engines, slot count and opcode match, and every non-address, non-reserved field is identical
(0 differences in 93 + 110 + 80 + 7 + 10 slots). Only address fields differ, which is expected
because the two cores use different buffers. The edit was applied symmetrically.

Consistent with the commit message? The HEAD commit says "ACT_TABLE_LOAD 22 -> 6, arithmetic
unchanged". This iteration has no commit message or note. The diff shows more than a table-load
change: it is a large rebatching of the ACT loop, with some of it moved to DVE. Because the
arithmetic is claimed unchanged, the cheap check is to compare outputs bit for bit against HEAD,
not only against the golden. verify reports `max_abs_err 2.322e-02` for output_0 and `1.633e-03`
for output_1. Both are identical to the baseline's verify, to four digits. That is
consistent with bit-exact numerics, but nobody has compared bits.

## 2. Is the latency change real?

Yes. `profile_latest.json`: median 44.612 us, samples 44.495-45.130, spread 1.42%. My re-run
gave 44.837. Baseline is 76.054, spread 0.54%. HEAD was 55.6. The gain over HEAD is about
11 us (20%) and the gain over baseline is 31.4 us (41%). Both are ten or more times the spread
of about 0.6 us, so this is not noise. Note that the 1.42% spread is larger than the baseline's
0.54%, and one sample (45.13) is an outlier. Any candidate below about 0.7 us cannot be
resolved with `--repeats 5`. Use `--repeats 10` or more for small edits.

## 3. Safety beyond the verifier

* **Semaphores.** `semaphores.py` on both subgraphs: for sems 4, 5 and 6, max_wait == total_inc
  (63/63, 38/38, 66/66), identical on sg00 and sg01. DMA sems 9-16 and 24-27 are 16/16.
  Sem 8 has 16 increments and no waiter, and was already like that. No wait exceeds its
  producer total, so nothing deadlocks. Nothing was dropped to get there: the waits were
  re-based, not removed.
* **Looseness.** Waits that stay valid are not what a single input set can falsify.
  Re-batched ACT/DVE code with private SBUF buffers may have WAR hazards (a buffer rewritten
  before its last reader finishes). That passes with one fixed input if the timing happens to
  be lucky. Mitigation: run `verify.py` 3-5 times, since the profile showed stable engine
  timing, and keep the run-to-run output identical.
* **The two cores** run the same instruction schedule, so they cannot disagree on structure.
* **Error margin.** `max_abs_err` is 86% of the 0.027 limit for output_0. That is the
  baseline's own value, so the iteration did not move it, but there is little headroom for any
  edit that changes rounding (bf16 vs fp32 intermediates, replacing a function). Treat
  numeric-changing candidates as high risk.

## 4. Candidates for the next iteration

State of the schedule (sg00; sg01 is the same within about 0.3 us). Profile timeline is about
59.8 us because it includes the profiler's instrumentation; `total_exec_time` is 44.8 us.
* **Loop window** (profile t = 29-55 us): Scalar 19.8 us busy, Vector 17.3, PE 12.9, Pool 1.3.
  ACT is the critical engine at 89% busy overall. It still idles 0.3-1.2 us before each of 5
  chunks, waiting on PE/DVE.
* **Head window** (t = 13-29): Scalar 10.8 of 16 us. Six `ACT_TABLE_LOAD` at 1.283 us each
  (7.7 us), one wide silu at 1.48, one wide CAST at 1.47. Two DVE `TENSOR_TENSOR` of 1.75 us
  each (1536 elements) sit back to back.
* Scalar busy by opcode: CAST 8.5 us (24 SBUF->SBUF 128-elem at 0.29 + the wide one),
  ACTIVATE 9.1, ACT_TABLE_LOAD 7.7, READ_ACCUMULATOR 3.5, COPY (PSUM->SBUF) 3.0.
* **Pool is idle** in the loop (22% busy overall, almost all of it DGE descriptor work).

| # | Candidate | Where | Change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Move the 24 SBUF->SBUF 128-elem `CAST` (FP32->BF16) from ACT to Pool, one head chain first | sg00+sg01, `Activation0.bin` slots 19-21, 25, 27, 28, 30, 33, 43, 44, 46, 47, ... (all `CAST` with src marker 0); insert `CAST` or `TENSOR_COPY` slots in `Pool0.bin` | Old ACT slot becomes `EVENT_SEMAPHORE` keeping ACT's sem-4 count; new Pool slot waits on the producer's sem and increments a Pool semaphore; consumers re-point to it. `CAST`/`COPY` run on Pool (docs ch. 14.3 item 3; `generated/opcodes.md`). Mechanics "verified", speed on this kernel "measure". | ACT CAST total 8.46 us per core, 24 x 0.29 = 7 us of it. Bounded by Pool op cost (unmeasured) plus one extra semaphore hop on a latency-bound chain. Realistic 1-3 us. Ceiling about 7 us only if the chain stops being latency-bound. | New semaphore on Pool; wrong re-pointing means a race the verifier may miss (see section 3). | Move **one** CAST only (the wide 1536 one in the head, slot 13, 1.47 us ACT; its input arrives via sem 6 val 11). Read its Pool `duration_ns` in the next profile. If Pool takes more than 1.5 us, or total does not drop beyond 0.7 us, kill it. |
| 2 | Split the two back-to-back 1.75 us DVE `TENSOR_TENSOR` (1536 elem) across DVE and Pool | `DVE0.bin` slots 24 and 25 (both sgs); slot 24 is FP32 MULT [128,12], waits sem 4 val 6; slot 25 is BF16 MULT, waits sem 13 val 16 | Move slot 25 to Pool (`TENSOR_TENSOR_ARITH_OP` runs on Pool) with an `EVENT_SEMAPHORE` placeholder on DVE | The two ops have different producers (visible in the waits), so they look independent. Same mechanism as 1. | At most 1.75 us on the head's serial chain. If Pool runs a 1536-elem op at over 2x DVE it gains nothing. | Low-moderate. Output goes wrong if the dependency is misjudged; verifier would catch it. | Check that slot 25 does not read slot 24's output (compare their src/dst addresses), then measure Pool duration for that one op. |
| 3 | Remove the second `ACT_TABLE_LOAD` of table 2 (rsqrt) by moving the head `ACTIVATE` func 29 (slot 18) next to slot 7 | `Activation0.bin` slots 6-18, both sgs | Reorder slot 18 (and its dependants) before the t0/t3 loads, delete slot 17, rebase ACT sem counts that follow. Each function lives in its own table set (`activation_functions.md`: 5 sigmoid, 7 exp, 9 softplus, 29 rsqrt, 36 silu are each in one set only), so five loads is the floor and six is current. Func 30 (square) is in many sets, so it is free. | Exactly 1.283 us on ACT, and the head ACT chain after the silu is serial, so this probably transfers to the total. | Medium. Slot 18's inputs: it waits only on PE sem 5 val 4, but PE matmul 4 may depend on an ACT result from slots 9/10/14. If so the reorder is illegal. | Walk the dependency of PE matmul 4 (what it waits on and which ACT output it reads) before editing. If it needs f7 or f5 outputs, kill it. |
| 4 | Fill the 1.7 us ACT stall at the start of the head (ACT is idle from 14.1 to 15.8 waiting for input DMA, sem 15) | `Activation0.bin` slots 1-3 | Issue an independent table load or work in that gap, then run the silu | Table load for the next set could overlap the DMA wait, but the order is fixed by data: t4 (silu) is needed first. Alternative: pull the DMA earlier (see unknown 1). | Up to 1.28 us (one load) if one later load can be placed inside the wait. | Low: pure ordering. | Check which later ACT instruction has no dependency on the silu output and can precede it. Likely only the t1 load itself, so the gain is the load time. |
| 5 | Move the 12 per-head rsqrt `ACTIVATE` func 29 [1 elem] off ACT | `Activation0.bin` slots 26, 34, 41, 45, ... (SBUF->SBUF, 1 elem, 0.2 us each) plus `READ_ACCUMULATOR` (12 x 0.295 = 3.5 us) | Replace with a DVE/Pool `TENSOR_SCALAR` pow(-0.5)-type op | `TENSOR_SCALAR_ARITH_OP` runs on all three engines (ch. 14.3). Does a pow ALU op exist, and does it match rsqrt exactly? **Not established.** | 12 x 0.2 = 2.4 us on ACT; 3.5 us more if the accumulate+read pair is also replaced. | **High**: changes rounding (output_0 is at 86% of the limit) and moves work onto DVE which is already 75% busy. | Compute the golden with the pow op in torch float32 and compare to rsqrt first. Do not touch the device until that agrees to below 1e-6. |
| 6 | Software-pipeline the loop so PE/DVE results for the next chunk are ready before ACT needs them | `PE0.bin` order, 5 chunk starts | Reorder PE matmuls and re-point waits; ACT gaps before the five chunks are 0.6-1.2 us each | Reorder mechanics verified (r08). | The 5 gaps before each chunk's first CAST sum to about 3.7 us in the profile. Part of it is irreducible latency. | Medium-high; a lot of wait re-pointing. | Look at which engine each stalled CAST actually waits on (`evt_wait_time_ns` and semaphore index). If it waits on DVE, PE reorder will not help. |

Ranking by gain per risk and effort: 1 (start with the single wide CAST) first because it
tests the Pool-speed question that candidates 1 and 2 both depend on; then 2; then 3 (a sure
1.28 us if the dependency check passes). 4 is mostly a subset of 3. 5 and 6 are last.

### Closed
* **Silu activations as separate per-head ops** (22 table loads): hoisted into one wide ACTIVATE,
  76.0 -> 55.6 us (HEAD commit).
* **The ACT instruction count / per-head rebatching**: 385 -> 238 ACT instructions took
  55.6 -> 44.6 us (this iteration, with DVE +15).
* **DMA as the bound**: DMA is 19 us busy (42.6% of exec), HBM traffic unchanged. Do not chase
  DMA volume.
* **PE as the bound**: PE busy is 19.2 us and 12.9 us in the loop. Not critical.
* **`MATMUL.perf_opt`**: not a lever (playbook ch. 14.3).
* **Removing a wait on an engine's own completion semaphore**: verified negative (ch. 14.2).

### Unknown
1. **What gates the input DMA arrival at about 15.8 us profile time?** Settle it by reading
   `DmaPacket.parquet` for queue completion times against the Pool DGE descriptor issue times
   (Pool's first instruction is at 6-7.7 us). If SP/Pool triggers can issue earlier, the whole
   head moves earlier, which could be worth several us.
2. **Pool instruction cost for a 128-elem and a 1536-elem CAST/TENSOR_TENSOR.** One trial
   instruction answers it. It decides candidates 1 and 2.
3. **Which engine each loop-chunk stall waits on.** Group `evt_wait_time_ns` by wait semaphore
   index in `Instruction.parquet`; decides candidate 6.
4. **Is this iteration bit-exact to HEAD?** Dump both outputs and compare. If not exactly equal,
   some arithmetic changed and the error margin (86%) needs a second look.
5. **Run-to-run output stability** over several `verify.py` runs, to test for a lucky-timing
   WAR race in the rebatched buffers.
6. **Whether the remainder is irreducible at the NEFF level.** The loop is about 90% ACT-busy
   with fixed per-instruction overhead (about 0.2 us per op). I could not establish an absolute
   floor, so the list above is not exhaustive and its ceilings are upper bounds, not forecasts.
