# Review of iter-1 (selective_scan b1 c2560 s4096 n16 bf16)

## 1. What changed in the NEFF
`neff_diff` shows 3 members changed. The other 31 are identical.

| member | change |
|---|---|
| `sg00/Pool0.bin` | 3842 → 6056 slots (+2214). It now runs 2214 `TENSOR_TENSOR_ARITH_OP` (fp32 `ADD`, 512 elems × 128 partitions, `acc += term`). |
| `sg00/DVE0.bin` | Slot count is unchanged (15929). 4439 slots differ. The moved `TENSOR_TENSOR_ARITH_OP` slots became `EVENT_SEMAPHORE` placeholders, and some of them carry the wait on Pool (`S[3]>=N`). |
| `sg00/Activation0.bin` | 252 slots differ. Only `events_semaphore_value` changed (e.g. 14→27, 30→58, 93→176). These are re-based waits on DVE's semaphore 6. |

- **Semaphore accounting:** `semaphores.py` on both unpacked NEFFs differs only in two rows.
  - Semaphore 3 totals 2488 → 4702. It is now incremented by Pool COPY (2488) plus Pool TT (2214).
  - Semaphore 6 waiters went 17683 → 19897. Its totals are unchanged at 8525.
- **Notes:** `notes/README.md` is still the empty template. The agent wrote nothing, so there is no claim to check the diff against.
  - The diff itself is coherent: it moves the accumulate-ADDs from DVE to Pool and re-bases the waits.
  - The agent should record this as `notes/dve_to_pool_add.md`.

## 2. Is the latency change real?
Yes. The numbers:

| run | median | min–max | spread |
|---|---|---|---|
| baseline (5 runs) | 7919.96 µs | 7918.8–7920.7 | 0.024% |
| iter-1 (5 runs) | 6886.38 µs | 6886.16–6886.48 | 0.005% |
| my rerun, 1 repeat | 6885.99 µs | | |

- The gain is −1033.6 µs (−13.05%), about 40,000× the spread, so it is not noise.
- Engine busy confirms the mechanism:
  - DVE 7860 → 6607 µs.
  - Pool 1214 → 5579 µs.
  - Everything else is unchanged.
- Pool instruction count went 3866 → 6080, which is +2214 as expected.

## 3. Safety
- **Numerics:** `max_abs_err` is 0.125 (83% of the 0.15 limit) on output_0, and 1.8e-4 on output_1. Both are identical to the baseline's verify numbers.
  - Same fp32 ADD, same order, so the error did not move.
  - The 83% is inherited from the baseline, not caused by this edit.
- **Semaphores:** per-semaphore total increments are conserved. Max-wait values stay ≤ total increments.
- **Remaining race risk (the verifier cannot see it):** Pool now consumes the `term` buffers (0x7940, 0xa940, …) and the accumulator (0x4940).
  - DVE must not overwrite a term buffer before Pool's ADD has read it. This is a WAR hazard, and the placeholder `S[3]>=N` waits are meant to cover it.
  - A mis-based wait would pass on a fixed input and could still race.
  - Pool instruction durations vary (1.27 µs vs 2.16 µs under contention), which is the timing jitter that exposes such a race.
  - Cheapest check: run `verify.py` with `--runs` greater than 1. Also run it on a second random input set. Also diff the per-buffer (reader, writer) wait chain once.
- **Pool ADD vs Pool COPY contention:** Pool COPY median went 183 → 636 ns (sum 766 → 1760 µs). Pool is now 81% busy and DVE is 99% occupied.

## 4. Where the time is now (new profile, `/tmp/pnew`)
- DVE instruction-union busy is 6846 µs of a 6894 µs span, so DVE is ~99% occupied and is still the critical path.
- DVE contents:
  - 2560 `TENSOR_TENSOR_SCAN` at 1.27 µs each (DVE only, cannot move).
  - 5120 fp32 `MULTIPLY` at 0.67 µs. Every one has a PSUM operand (`0x2xxxxxx`), so Pool most likely cannot take them.
  - 506 fp32 `ADD` (708 ns) and 160 bf16 `MULTIPLY` (703 ns). These are SBUF-only and still on DVE.
  - 191 `EVENT_SEMAPHORE` stalls longer than 200 ns, totalling 438 µs. They are waits on Pool (`S[3]`). The baseline had 16, totalling 19 µs. Moving the work cost some DVE stall.
- Realised gain was 0.47 µs of wall time per ADD moved, against 0.68 µs of DVE instruction time. The difference is the new Pool-wait stalls.
- Instruction durations on DVE overlap (their sum is larger than the wall), so use wall deltas, not duration sums, for ceilings.

## 5. Ranked candidates for iter-2

| # | Candidate | Engine / member / slots | Change | Mechanism | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Move the rest of the SBUF-only TT ops DVE→Pool | `DVE0.bin` and `Pool0.bin`. DVE `TENSOR_TENSOR_ARITH_OP` with no PSUM operand: 506 fp32 ADD and 160 bf16 MULTIPLY. Find them via the profile `operands` field with no `0x2xxxxxx` address. | Same recipe as iter-1: copy the slot into Pool, turn the DVE slot into `EVENT_SEMAPHORE`, preserve increments, re-base waits (Activation `events_semaphore_value` and PE waits). | Established on device this iteration: 2214 ADDs moved gave −1034 µs. | 666 ops × 0.47 µs ≈ **310 µs**, realistic 100–250 µs. Pool headroom: 6886−5579 = 1307 µs, and each moved op costs Pool about 1.3–2.2 µs, so Pool saturates after about 500 ops. | Same WAR-hazard class as iter-1, and the verifier can pass it by luck. | Move 100 ops. If the wall gain is under about 0.3 µs/op, or Pool becomes the critical path, stop. |
| 2 | Cut the DVE stalls on Pool by reordering the Pool stream | `Pool0.bin` | Hoist the carry `COPY`s (the ones DVE's scan waits on, `S[3]>=589` etc.) ahead of the slow Pool ADDs. They are in-order behind them today. | Pool is an in-order queue. The 191 DVE stalls over 200 ns come from the scan waiting on a COPY queued behind 1.3–2.2 µs ADDs. | Bounded by the 438 µs total stall time. Realistically 100–250 µs, and it overlaps with #1. | A reorder can break RAW/WAW on the accumulator buffers. Waits are counting semaphores, so a swap must keep the sem-3 increment order that consumers count on. | Locate the 191 stalls with `profile.py --parquet`. Check whether the stalled waits point at a COPY or an ADD, and whether a COPY precedes an ADD in program order. If they are all ADDs, drop this. |
| 3 | Drop the carry COPY by pointing the scan `imm` pointer at the previous scan's last column | `DVE0.bin` `TENSOR_TENSOR_SCAN` (2560 slots), `Pool0.bin` `COPY` (2560 slots) | Set `imm` to the previous scan's last-column address (`0x713c`, `0xa13c`, …) instead of the copy target (`0x4110…`). Turn the COPY into a bare sem-incrementing no-op so counts hold. | The profile shows COPY `src=0x713c` (previous scan's last element) `dst=0x4110…` (scan imm). | No direct DVE gain. It frees Pool COPY time (1760 µs now, 183 ns uncontended) so #1 can go further. Unknown beyond that. | The scan may read its `imm` while the output buffer is being rewritten (same address). The ISA semantic of `imm` is unverified. | Test one scan on a single partition-state offline (`run_neff`), or check the `isa.py` doc for when `imm` is read. If it is not read before the first write, drop it. |
| 4 | Offload PSUM MULTIPLYs: ACT copies PSUM→SBUF, Pool multiplies | `Activation0.bin`, `Pool0.bin`, `DVE0.bin` | For a subset of the 5120 PSUM MULTIPLYs, add an ACT copy and a Pool multiply. | ACT is 27.6% busy, so there is headroom. Whether Pool can read PSUM at all is unverified. | Unknown. ACT 723 ns + Pool 1268 ns per op vs DVE 671 ns. It needs spare slots, which the NEFF does not have. | High: it adds instructions and buffers, the SBUF is already allocated, and any allocation change can pass one input and fail another. | Check the ISA/NEFF docs for Pool PSUM access. If Pool reads PSUM, move just 50 ops and measure. Otherwise drop. |

## Closed
- **Moving the fp32 accumulate ADDs (2214 of them) to Pool:** done, −1034 µs. Do not redo.
- **Moving scan or PSUM MULTIPLY to Pool:** not supported. Scan is DVE-only in the ISA, and the MULTIPLYs read PSUM. This is evidence from the profile, not a measured refutation.
- **Using bf16 instead of fp32 for DVE TT throughput:** the bf16 MULTIPLY (703 ns) is not faster than fp32 (671 ns). Closed. Also `max_abs_err` is already 83% of the limit, so there is no precision budget to trade.
- **Re-measuring noise:** spread is 0.005%, so single-run comparisons are trustworthy.

## Unknown
- Whether Pool can read PSUM. This decides #4. Check the ISA doc, or try one Pool op with a PSUM operand.
- The `imm`-pointer read time in `TENSOR_TENSOR_SCAN`. It decides #3.
- Whether the iter-1 WAR waits are complete. Settle it with `verify.py --runs` set to 5 or more, plus a second input set.
- What the 191 DVE stalls wait on (COPY vs ADD). It decides #2.
- Whether the irreducible DVE floor is about 6.7 ms. The scan (3.3 ms) plus the PSUM MULTIPLYs (3.4 ms) cannot leave DVE. If they stay, the whole remaining headroom is roughly 100–300 µs (about 1.5–4%).
