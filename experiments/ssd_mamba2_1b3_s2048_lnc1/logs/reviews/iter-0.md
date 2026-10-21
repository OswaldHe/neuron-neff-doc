# Review: iter-0 (ssd_mamba2_1b3_s2048)

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports: header identical, 37/37 members identical, "payloads are identical". `notes/` holds only the template README, with no attempt recorded. `profile_latest.json` and `verify_latest.json` do not exist in the worktree. There is no claim to check against the diff.

## 2. Is the latency change real?
There is no change. I re-profiled the unedited NEFF (`profile.py --repeats 1`): 7157.334 us against the baseline median of 7154.946 us (+0.03%). The baseline's own spread is 0.021% (7154.875 to 7156.367 us, about 1.5 us), so the 2.4 us difference is inside the noise. Anything under about 5 us on this kernel is noise. The kernel is long and very repeatable, so a real win of 20 us or more is detectable with `--repeats 3`.

## 3. Safety
There is nothing to check, because no edit was made. For reference, the baseline verifier error is `max_abs_err` 5.122e-07 against a limit of 6.2e-07, which is 83% of the limit. The headroom is only about 1e-7, so any change to numerics (candidates 1 and 2 below) must be re-verified. Run `semaphores.py` on the unpacked tree after every structural edit.

## 4. Where the time goes
Engine busy times, taken from `profile_baseline.json` and the per-instruction table.

| Engine | Busy | Share of 7155 us |
|---|---|---|
| Activation (ACT) | 6284 us | 87.8% |
| DVE | 2733 us | 38% |
| Pool | 2072 us | 29% |
| PE | 1967 us | 27% |
| DMA | 1879 us | 26% |
| SP | 802 us | 11% |

ACT is the bottleneck engine. Its time is made up as follows:

- **`ACT_TABLE_LOAD`:** 2046 slots x 1283 ns each = **2625 us (37% of the whole runtime)**.
- **`ACTIVATE`:** 10240 slots, 3841 us. Of these, 7168 are `COPY` (func 128, about 430 ns each, 7 per head-chunk) and 2048 are exp (func 7).
- **`ACTIVATE` reciprocals:** 1024 `ACTIVATE` slots with func 129 (reciprocal).
- **`DMA_DIRECT2D` triggers issued from ACT:** 1088 slots, 609 us. 1024 of them are at `ssd_block.py:275`.

Why the table loads happen:

- `Activation0.json` loads two table sets: `natural_log_exp_and_others` (sel 0) and `reciprocal_and_small` (sel 1).
- The profiler's names "SOFTPLUS" and "UNKNOWN" are wrong. The "SOFTPLUS" slots are func 7 (exp, set 0) and the "UNKNOWN" slots are func 129 (reciprocal, set 1).
- Each of the 1024 (head, chunk) units issues `exp(cs_last)` at `ssd_block.py:226` and `exp(cs_col)` at `ssd_utils.py:117`. Both use set 0.
- The unit then issues `reciprocal(exp_cs_col)` at `ssd_utils.py:119`, which is `exp(-cs)` computed as 1/exp(cs). This needs set 1.
- The result is two reloads per unit: sel 1 before the reciprocal and sel 0 after it. That gives 1023 loads of each set.
- The reciprocal operates on a [128,1] tile. It does 294 ns of work but costs 2 x 1283 ns of reloads.

## 5. Ranked candidates for iter-1

### Candidate 1 (best): replace the ACT reciprocal with `exp(-1 * cs_col)` and delete the table loads
- **Where:** Activation0.bin only. In each unit, the func-129 `ACTIVATE` at `ssd_utils.py:119` (1024 slots; the first is slot 9, then 17, 40, ...) is edited. The two `ACTIVATION_TABLE_LOAD` slots around it are removed: the sel=1 load before it and the following sel=0 load (the one attributed to `ssd_block.py:226`).
- **What changes in the reciprocal slot:**
  - `activation_func` goes from 129 to 7.
  - `scale_value_imm_fp32` goes from 1.0 to -1.0.
  - The source start address is re-pointed from `exp_cs_col` (the dst of the previous exp) to `cs_col`, which is the source of the previous exp slot. Copy that address from the preceding func-7 slot, which has the same shape [128,1].
  - The dst stays the same.
- **How to remove a load slot:** turn it into a no-op that keeps the stream layout (a NOP or an `EVENT_SEMAPHORE` with no wait or update). The loads have `events_wait_mode=NONE` and `events_update_mode=NONE`, so no semaphore count changes. Check this per slot before editing.
- **Mechanism:**
  - `exp(-x)` is mathematically identical to 1/exp(x).
  - Set 0 contains exp (`act_info.json`), so no reload is needed.
  - The "edit slot fields + NOP out a table load" mechanics are Verified (ch. 13, ch. 14.2).
  - Using a different function id inside an already-loaded set is plain ISA (ch. 7.2).
  - The ISA guarantees activation function ids.
  - Note that `activation_func` is the function id and the set must be active (ch. 7.2).
  - The "table_sel" semantics are only Observed, not tested by editing (ch. 3.5), so verify that deleting a load really leaves set 0 active.
- **Expected gain:** up to 2625 us of ACT busy time removed, so ACT busy drops from 6284 to about 3660 us. Wall time is bounded below by the next engine, which is DVE at 2733 us, and by dependency chains, which are unknown. A realistic result is somewhere between 1.0 and 2.5 ms (about 15 to 35%). The ceiling is the 2625 us of load time, and the load is serialised on ACT, so it is not hidden behind other engines.
- **Risk:**
  - The numbers change slightly. The reciprocal table and the exp table have similar piecewise-polynomial accuracy, but the error could move either way.
  - Baseline error is at 83% of the limit, so the verifier is the check. It is a fixed-input check, but the change is a value change, not a race, so one verified input set is adequate evidence.
  - The synchronisation is unchanged. The new slot only reads `cs_col`, which was produced before the exp slot, and the old wait on the exp's semaphore is conservative.
  - Overflow behaviour for very negative `cs` is the same domain as before.
- **Cheapest falsifier:**
  1. Edit just the first unit, re-pack, and run `verify.py`. If `max_abs_err` leaves the limit, stop.
  2. Then do all 1024 units and run `profile.py`. The ACT busy time should drop by about 2.6 ms.
  - If the error approaches 6.2e-7 only with the exp-based edit, try Candidate 2.

### Candidate 2 (fallback if error is too close): move the reciprocal to DVE or Pool
- **Where:** the same 1024 reciprocal slots in Activation0.bin, plus a new slot in DVE0.bin (or Pool0.bin). The table loads are removed as in Candidate 1.
- **What changes:**
  - The ACT slot is replaced by an `EVENT_SEMAPHORE` that keeps ACT's semaphore count (ch. 14.3 item 3, ch. 10.4).
  - A DVE `TENSOR_SCALAR` divide, or a reciprocal opcode if one exists, is inserted with a wait on the exp's ACT semaphore.
  - The consumers of `exp_neg_cs_col` are re-pointed to wait on the DVE semaphore.
- **Mechanism:** move-between-engines is Verified for mechanics only (ch. 14.3 item 3; no measured saving). An exact DVE divide should be at least as accurate as the table.
- **Expected gain:** the same ceiling as Candidate 1, at most 2625 us of ACT load time, minus the added DVE work (1024 small instructions, under 0.3 ms) and the new cross-engine latency, which is unknown.
- **Risk:** higher than Candidate 1. This is a synchronisation change and the verifier can pass it once and still race. Run `semaphores.py`.
- **Cheapest falsifier:** check that DVE has an instruction that can compute 1/x. If it does not, drop this candidate.

### Candidate 3: move ACT-issued DMA triggers to the under-used SP queue
- **Where:** Activation0.bin `DMA_DIRECT2D` slots at `ssd_block.py:275` (1024 slots, 572 us) and the others (1088 total, 609 us). SP is only 11% busy.
- **What changes:** the trigger slot moves to the SP stream with the same queue and semaphore fields, and ACT keeps an `EVENT_SEMAPHORE` placeholder if it owns a count.
- **Mechanism:** "re-point or reshape a queued DMA through `SP0.json`" is Verified (ch. 14.2, r09), but moving a trigger between engines has not been established. Check ch. 9.
- **Expected gain:** at most about 570 us of ACT time, which only helps once Candidate 1 has made ACT stop being the bottleneck. Without Candidate 1 it probably shows up as a partial gain.
- **Risk:** medium. DMA queue and semaphore ownership can break, and this is a race-prone change.
- **Cheapest falsifier:** before editing, read the SP stream and the queue ownership for those triggers. If the queue is bound to an engine, drop it.

### Candidate 4: after Candidate 1, rebalance the 7 per-unit ACT COPY activations
- **Where:** the `ACTIVATE` slots with func 128 (7168 slots, about 430 ns each, 3.1 ms) at `ssd_block.py:454/463/477/504/514/535` and `ssd_utils.py:100`.
- **What changes:** move some of them to Pool, which is 29% busy, or to DVE.
- **Mechanism:** Verified mechanics (ch. 14.3 item 3); engine speed for these specific copies is unmeasured (ch. 14.3 says "measure").
- **Expected gain:** unknown. It would only matter if ACT is still the bottleneck after Candidate 1.
- **Risk:** high, with many synchronisation edits.
- **Cheapest falsifier:** profile after Candidate 1. If ACT busy is below DVE or PE busy plus its idle time, skip this.

## Closed
No lever is refuted by measurement for this kernel yet: the notes are empty. From `neff_doc` ch. 14.2, other kernels showed that removing waits on an engine's own completion semaphore gave wrong output (DVE) or out-of-bounds errors (Pool); do not loosen those waits.

## Unknown (measure first)
- Whether deleting a table-load slot leaves the intended set active on this device. Settled by running Candidate 1 on a single unit and checking the output.
- How much of the 2625 us of load time is on the end-to-end critical path. Settled by running Candidate 1 across all units and measuring wall time.
- Which engines have an instruction that can compute 1/x (needed for Candidate 2).
- The accuracy of exp(-x) against 1/exp(x) relative to the 6.2e-7 limit. Settled by `verify.py` on the edit.
