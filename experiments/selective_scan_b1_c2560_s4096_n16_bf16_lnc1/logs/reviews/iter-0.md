# Review, iter-0 (selective_scan b1 c2560 s4096 n16 bf16)

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports identical header, 34/34 members identical, "payloads are identical". There is no `profile_latest.json`, no `verify_latest.json`, and `notes/` holds only the template README. So there is no edit to compare against any notes claim.

## 2. Is the latency change real?
There is no change. I re-profiled the unedited NEFF as a reference: 7919.3 us, against a baseline median of 7919.96 us (min 7918.78, max 7920.68, spread 0.024%). Run-to-run noise is about +-1 us (about 0.01%), so any edit that gains more than about 5 us is distinguishable from noise. This kernel is quiet.

## 3. Safety
Nothing was edited, so no synchronisation was loosened. Baseline error is already max_abs_err 0.125 against a limit of 0.15 (83%, cosine 0.999944). Any edit that changes numerics, for example bf16 intermediates or reassociating the accumulation, has about 17% headroom and probably fails. Prefer bit-exact edits.

## 4. Where the time goes (baseline profile, per-instruction parquet)
The kernel is 20 channel-tiles x 16 states x 8 chunks of 512 = 2560 units. Each unit runs this DVE chain (source lines are in vendor `scan/selective_scan.py`):

| DVE op | line | count | mean ns | total ms |
|---|---|---|---|---|
| TT MULTIPLY `dtx*B_full` (B_full read straight from PSUM) | 224 | 2560 | 663 | 1.70 |
| TENSOR_TENSOR_SCAN mul,add | 233 | 2560 | 1275 | 3.26 |
| TT MULTIPLY `C_full*state` | 267 | 2560 | 683 | 1.75 |
| TT ADD `y_accum += Cs` | 275 | 2560 | 683 | 1.75 |
| 160+160 small TT (lines 182, 297), 200 MEMSET, 5176 EVENT_SEMAPHORE | | | | 0.11 + 0.08 + 0.20 |

- DVE is 99.2% busy (7860 us of 7920 us). It is the critical engine.
- The other engines are far from busy: PE 37.5%, ACT 24%, Pool 15.3% (1214 us), DMA 18.8%.
- Pool does only 2560 one-element carry COPYs (301 ns each) and 680 DMA_DIRECT2D.
- ACT runs 2560 EXP-type ACTIVATEs at about 717 ns, plus 160 SOFTPLUS at 813 ns.
- PE runs 2 x 2560 K=1 broadcast matmuls at about 586 ns each.
- TT costs about 1.33 ns/element at fp32 (512 elements per partition, about 680 ns). The scan costs 2.5 ns/element.
- Consecutive DVE instructions have 5181 gaps >50 ns, summing to about 2.27 ms. DVE therefore also stalls on producers (ACT, PE, Pool), not only on its own work.

## 5. Ranked candidates

| # | Edit | Mechanism / source | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Move the line-275 ADD (`y_accum += Cs`) from DVE to Pool for a fraction of the 2560 units.** DVE stream: replace each moved slot with an `EVENT_SEMAPHORE` that keeps DVE's count. Pool stream: insert `TENSOR_TENSOR_ARITH_OP` ADD (opcode 0x41, Pool-capable per `generated/opcodes.md`). Re-point the consumers' waits (next ADD, final y store) at Pool's semaphore. | neff_doc 14.3 #3 and ch.10.4. Pool executes 0x41 (1814 occurrences in the corpus). | Ceiling is the 1.75 ms of line-275 DVE time (22%), if Pool took all of it. Pool has about 6.7 ms free, but its TT speed is unmeasured. Pool fixed overhead looks high (a 1-element COPY costs 301 ns). Realistic gain is a 25-50% share, so 0.4-0.9 ms. | Semaphore rewiring across many slots can race. The verifier uses one input set and would not catch a race, so run `semaphores.py` and repeat the verifier 3+ times. Numerics are bit-exact (same fp32 add). | Convert ONE unit's ADD to Pool and read its duration in the profile (Instruction.parquet). If Pool takes >1.4 us per 512-element fp32 add, shrink the share or drop the candidate. Total cost is minutes. |
| 2 | **Same move for the line-267 multiply `C_full*state`** (src1 is PSUM; Pool PSUM access is unverified). Do it after #1 shows Pool TT speed. | Same as #1. | Ceiling 1.75 ms. #1 and #2 together are bounded by Pool capacity (about 6.7 ms free). If Pool TT is 1.4 us, about 4.8 ms of work fits, so split about 2:1 DVE:Pool. | As #1, plus Pool reading PSUM may be disallowed or slow. | A single-slot test of the PSUM operand on Pool. |
| 3 | **Re-balance the line-224 multiply** (`dtx*B_full`, 1.70 ms) to Pool in the units not used by #1/#2. | Same mechanism. | Ceiling 1.70 ms. Competes with #1/#2 for the same Pool budget; use whichever shows the best Pool throughput. | Same as #1. | Same as #1. |
| 4 | **Replace the ADD accumulation by PE PSUM accumulation** (identity stationary, `acc_flags` accumulate across the 16 states). Only worth trying if fp32 matmul is cheap. | PE busy is only 37.5%. Existing matmuls use acc_flags=3 already. | Ceiling 1.75 ms. fp32 matmul rate is unmeasured. bf16 moving data would cut cost 4x but breaks the error budget (83% used). | High: structural, changes PSUM allocation, and fp32 PE may cost >2 us x 2560. | Time one fp32 identity matmul of 512 columns. If it is >1.2 us, drop. |
| 5 | **Trim DVE EVENT_SEMAPHORE slots (5176 x 24-54 ns = 202 us) and move the 200 MEMSETs (82 us) to Pool.** | Ch.10 rules: only drop waits already implied by another wait on the same semaphore. | Ceiling about 0.28 ms (3.5%). Realistic is much less. | Highest sync risk per us gained, and the verifier would not detect a race. | Only the 16 stalls >100 ns (18 us total) are real waits. If removing the 24 ns slots does not show up in the DVE span, drop. |

Recommended order: #1, tuned for share; then #2/#3 filling the remaining Pool budget; #4 and #5 only if Pool throughput is poor.

## 6. Closed
No prior iteration or notes exist, so nothing is closed by measurement. For reference:
- The scan (3.26 ms, 41%) is a single hardware op; I know of no edit that speeds it without changing numerics.
- The DMA (18.8%), ACT (24%) and PE (37.5%) lanes are not bottlenecks. Speeding them up alone cannot help.

## 7. Unknown (measure first)
- **Pool TT throughput and fixed overhead** for fp32 [128x512]. This decides #1-#3. Measure by converting one unit.
- **DVE perf-mode for 16-bit operands** (2x/4x, `v3-2x-4x-dve-perf-mode`). Which dtype combos speed up TT is unmeasured, the `performance_mode` column is empty, and any dtype change risks the 0.15 limit.
- **Whether the 2.27 ms of DVE gaps is producer-bound** (ACT exp, PE broadcast, Pool carry copy, or DMA). If it is, re-ordering or deeper prefetch of those producers may give a gain independent of the engine move. Measure with `evt_wait_time_ns` and the semaphore each DVE slot waits on, grouped by producer.
