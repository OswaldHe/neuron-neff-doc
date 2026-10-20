# Review of iter-6 (uncommitted `kernel.neff` on top of 880982b)

Note: `notes/` holds only the README, so there is no agent write-up to compare against. The working-tree
`kernel.neff` differs from HEAD (880982b, 57.5 us), so the diff below is HEAD -> working tree, not
baseline -> working tree. HEAD's commit message describes the *previous* edit, not this one.

## 1. What changed (HEAD -> working tree)
4 members changed, 30 identical.
- **DVE0.bin**: 133 -> 136 slots. About 120 slots shifted or rewritten by re-sequencing.
  The three new slots are `TENSOR_SCALAR_ARITH_OP`. They increment the new semaphores 68 and 69
  (1 increment each, 1 waiter each).
- **SP0.bin**: 67 -> 69 slots, 24 slots differ. EVENT_SEMAPHORE / PSEUDO_DMATRIGGER pairs were
  reordered and one DGE (PSEUDO_DMA_DIRECT2D) was added.
- **Activation0.bin**: 121 slots, 5 differ.
  - Slots 11 and 16: `TENSOR_SCALAR_ARITH_OP` became EVENT_SEMAPHORE (the compute op was removed from ACT).
  - Slots 118 and 119: swapped.
  - Slot 120: the last DMA's `src_num_elem` and `dst_num_elem` went 32 -> 64.
- **Pool0.bin**: 3 slots changed. Two DGEs became EVENT_SEMAPHORE.
- Semaphore accounting, HEAD -> now:
  - Sems 17 and 18 now have their DGE source on SP instead of Pool.
  - Sem 21 has one DGE source (Activation) instead of two.
  - Sems 68 and 69 are new.
  - Waiter counts on sems 4 and 6 shift by +1 and -1.

I read this as a re-balance of DVE-side compute (some work moved off ACT onto DVE, apparently in
the form of the three new tensor-scalar slots) plus moving DGEs between the Pool and SP queues.
That is my inference from the opcode changes; there are no notes to confirm it.
I did not decode every shifted DVE slot.

## 2. Is the latency change real? Yes.
Interleaved on the same device:

| NEFF | median | range |
|---|---|---|
| HEAD | 57.534 us | 57.26-58.06 |
| working tree | 55.245 us | 55.03-55.60 |
| working tree, repeat | 55.202 us | 54.75-55.51 |

- The ranges do not overlap.
- The gain is about 2.3 us (-4.0%) against a spread of about 1.0-1.4%.
- Total against baseline: 67.525 -> 55.2 us (-18.2%).
- `profile_latest.json` was overwritten by my repeat runs. It now holds the working-tree NEFF
  at 55.202 us.

## 3. Safety
- `semaphores.py` on the unpacked NEFF: no "wait exceeds increments" flag. Every semaphore's
  max_wait <= total_inc. Sem 21 went 32/32 -> 16/16 and sems 68/69 are 1/1, so counts are
  consistent, not loosened.
- Verifier: output_0 max_abs_err 6.09e-5 and output_1 max_abs_err 4.494e-4 (83% of the 5.4e-4
  limit). Both are bit-identical to the baseline verify file, so no new error. The 83% is
  inherited, not introduced.
- HBM write bytes are 131072, equal to the two outputs. The widened final DMA does not
  over-write.
- The residual risk is a race that the single fixed input set cannot expose. Three things carry it:
  - the new semaphores 68 and 69;
  - the DVE re-sequencing;
  - DGE queue moves, where an earlier issue can pass a semaphore guard once and fail later.
- Re-running the verifier a few times, and in a different run order, is cheap insurance.

## 4. Candidates for the next iteration
These come from the per-instruction table of the working-tree NEFF (`/tmp/profA`). The profile
timeline runs 0-69.8 us. About 12.4 us at the start and about 4 us at the end are runtime
preamble and postamble. The kernel body is about 12.4 -> 65.8 us.

| # | Candidate | Change | Mechanism | Expected gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | Spread the 7 input-load DGEs off Pool | Pool0 DGEs pc 64, 66-71 issue back to back at 13.45 -> 17.9 us, 0.65-0.8 us each, serially. Re-host 3-4 of them on Sync (idle 13.2-16.4 us) and on ACT (idle 13.83-16.39 us after ACT_TABLE_LOAD). Keep each DGE's semaphore and queue fields, change only the host stream, with the slot order and `events_*` guards preserved. | Earlier commits showed DGE queue follows its semaphore, and a DGE costs about 0.65 us of issue time on its host engine. Parallel hosts mean parallel issue. | Up to about 2 us. Bounded by first PE matmul at 15.98 us (about 3.4 us after the first Sync load at 12.41) and the later loads at 17.1 and 17.9 us. Real gain depends on which load gates the first consumers. | Low. Reordering of issue only. A missing guard would show as wrong inputs, and the verifier would likely catch it. | Move only DGE pc 71 (the last load) to Sync. If the first ACTIVATE at 16.39 us and the 22-25 us region do not move up, the loads are not gating. |
| 2 | Third host for the 5-DGE staging burst at 28-30 us | Sync 74/75/76 and Pool 76/77 issue serially (28.04 -> 29.95 us) while ACT sits in an EVENT_SEMAPHORE wait 27.9 -> 30.66 us. Host one or two of these DGEs on ACT, which is free in that window. | Same as #1. | Up to about 1.2 us. Bounded by the 2.8 us ACT stall, of which about 1.3 us is irreducible DMA latency. | Medium. ACT's wait on those semaphores must stay consistent (sem 4 counts). The DGE count semantics are checked by `semaphores.py`. | After the edit, ACT's CAST at about 30.8 us should start by about 29.6 us. If it does not, the stall is the DMA latency, not the issue. |
| 3 | Move the three back-to-back ACT TENSOR_SCALAR ops (pc 86-88, 23.8 -> 24.8 us) to DVE | DVE is idle 23.35 -> 25.38 us while ACT runs these three ops at 0.29-0.38 us each. Re-emit them in the DVE stream as `TENSOR_SCALAR_ARITH_OP`, and rewrite the semaphore increments and waits from sem 4 (ACT) to sem 6 (DVE). | ACT TS and DVE TS share the same S3D3_TS struct, and this iteration already did this move for slots 11 and 16. | At most about 0.7 us. Only if the three are independent. If they form a chain, the gain is about 0.1 us per op. | High. Needs a semaphore re-balance across both engines. A mis-count can pass once and race later. | Check whether op 87 reads op 86's output. If yes, drop the candidate. |
| 4 | Split the final output store across more queues | The final output_1 store (64 KB, Sync pc 130, 65.77 us) is followed by a 2.5 us wait to 68.3 us. Split it into 2-4 parts across Sync, ACT and Pool, as the previous iteration did for the other output. | A smaller transfer per queue completes sooner. | Unknown, perhaps 0.3-0.8 us. Bounded above by the 2.5 us issue-to-completion time. | Low. | Split in two and compare the completion time at about 68.3 us. |

Not listed: there is no candidate where the remaining critical chain (the 36-48 us region of
dense PE/DVE/ACT hand-offs) can be shortened by field edits alone. Its cross-engine hand-offs
are 0.1-0.3 us each and the work is a sequential recurrence.

**Closed** (from commit messages; `notes/` is empty):
- Spill and reload round trips through DRAM: fused to SBUF->SBUF DMAs, 67.525 -> 61.905 us.
- 4-to-1 DMA barriers: replaced with per-block waits.
- Already moved off ACT/SP in earlier iterations: the var9 reload DGEs and the staging DGE
  guards (61.8 -> 60.9 -> 59.4 -> 57.6 us).
- The first input load is already hosted on Sync.

**Unknown** (would change the ranking):
- Which of the 7 input loads gates which consumer. Settle it with a per-load dependency table
  from the DGE semaphore waits (or the falsifier for #1).
- DGE issue-time scaling with descriptor count (0.6 us for 2 KB, 0.8 us for 64 KB). Settle it
  by timing a DGE with fewer descriptors.
- Whether ACT ops 86-88 are independent (decides #3).
