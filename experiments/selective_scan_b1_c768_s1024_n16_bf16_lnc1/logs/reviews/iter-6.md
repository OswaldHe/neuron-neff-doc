# Review: iter-6 worktree (HEAD 7358c9d, "iteration 5" commit), selective_scan b1 c768 s1024 n16

## 1. What changed in the NEFF

I diffed `baseline/kernel.neff` against `kernel.neff`. That diff is cumulative, covering all five accepted iterations. Three members changed and 31 are identical:

| member | slots | change |
|---|---|---|
| `Pool0.bin` | 296 -> 389 | The 170 y+=Cs fp32 ADDs moved here from DVE, the carry COPYs were merged pairwise, and fresh ES hand-off slots were added. |
| `DVE0.bin` | 1204 -> 1055 | The ADDs became ES or MEMSET slots, and the 12 first-y writers became MULTs straight into y. |
| `Activation0.bin` | 234 -> 234 | 24 slots differ. ACT's `ACTIVATION_TABLE_LOAD` (slot 2 -> 1) is swapped with the ES that preceded it, and the dA `ACTIVATE` slots are re-ordered with their ES slots. |

The last commit's own diff against HEAD~1 is not readable with `neff_diff.py` through process substitution, so I went by the commit message. Its claims match the cumulative diff:
- Cs MULTs retargeted into y, with the dead COPYs dropped.
- The ACT table load hoisted. In slot 1, `ACTIVATION_TABLE_LOAD` takes the place of `EVENT_SEMAPHORE`, and the ES moves to slot 2.
- The dA `ACTIVATE` slots get new `events_wait_idx`, `events_update_mode` and `events_semaphore_value` fields. For example, `events_semaphore_value` 14 -> 22 on slot 26.
- Fields rewritten to move the output_1 carry array and the group-0 state tile in SBUF.

`notes/README.md` is the empty template. The agent wrote no notes, so there is nothing to cross-check beyond the commit messages.

## 2. Is the latency change real?

Yes.
- **Medians:** baseline 602.64 us (spread 0.05%) -> 501.44 us now. My own re-profile gave 501.27 us.
- **Spread:** `profile_latest.json` reports `spread_percent` 9.4%. That comes from one outlier sample of 548.41 us. The other four samples are 501.35 to 501.52 us, a 0.17 us range. The median is unaffected.
- **Last step:** 508.7 -> 501.4 us is 7 us against about 0.1 us of noise, so it is real. The 603 -> 501 cumulative gain is real.
- **Mechanism (the Instruction table):** DVE's union busy time is 502.2 us of a 513.8 us span, and DVE is 96% busy. The previous step cut DVE work and Pool-induced stalls.

## 3. Safety beyond the verifier

- **Error:** output_0 max_abs_err is 0.06245 (83% of the 0.075 limit). That is identical to the baseline value, and the notes say the output is bit-identical. Error has not moved.
- **Semaphore accounting:** I unpacked the NEFF and ran `semaphores.py`. No semaphore is flagged "wait exceeds increments". S[3] has 282 increments from Pool, S[4] has 194 from ACT, S[5] has 374 from PE, and S[6] has 640 from DVE.
- **Removed barriers:** `semaphores.py` checks counts, not placement. The iteration removed DVE barriers on the strength of "these SBUF ranges no longer alias". The verifier runs one fixed input set and cannot catch a WAR or RAW race that only shows up with a different timing.
  - The c2560 experiment's iter-7 notes show this is a real risk. The rotating region at 16384..60224 is shared across DVE, ACT and Pool, and an alias check done only within DVE missed a hazard (they hit it as `max_abs_err 43.7`).
  - The next agent should do a one-time five-stream alias audit of the three moved buffers (output_1 carry, group-0 state tile, and the ACT table load). Run the NEFF 20 or more times and compare outputs bit-for-bit as well.

## 4. Ranked candidates for the next iteration

**Model for this kernel.** The numbers come from `prof/Instruction.parquet`.
- There are 12 groups (6 channel tiles x 2 seq chunks of 512) x 16 states = 192 scans. DVE executes:

  | DVE work | count | each | total |
  |---|---|---|---|
  | `TENSOR_TENSOR_SCAN` | 192 | 1.27 us | 246 us |
  | fp32 MULTs (B-mult, C-mult) | 384 | 0.67 us | 257 us |
  | residual ADDs | 24 | 0.84 us | 20 us |
  | bf16 `dt*x` | 12 | 1.2 us | 15 us |

- Pool is 83% busy: 168 ADDs at 2.0 us each (338 us). PE is 43% and ACT is 29%.
- **Head:** the first DVE compute starts at 14.7 us. DVE's first slots are MEMSETs at 10.3 us, and nothing computes before 14.7 us.
- **Tail:** the last DVE compute ends at 509.1 us and the output DMA lands at about 512.3 us. After that is runtime teardown, which cannot be edited.
- The wall is therefore about DVE busy time + about 12 us.

The c2560 sibling experiment (`neff_doc/experiments/selective_scan_b1_c2560_s4096_n16_bf16_lnc1/logs/notes/iter-5..7.md`) measured the cost model for this same NEFF family. I use its numbers below. They are not re-measured on c768 unless said so.

**The honest summary:** the remaining headroom is small. It is probably 2 to 4% in total, and each candidate below is under 1.5%. Rank by gain per risk.

### Candidate A: measure the Pool-removal ceiling first (timing-only probe, no code change shipped)

| field | value |
|---|---|
| Engine and member | `Pool0.bin`, the 168 `TENSOR_TENSOR_ARITH_OP` ADD slots (512 fp32 elements). |
| Change | Set `num_elem` 512 -> 64 on all of them. The math is wrong, so this is a probe only. |
| Mechanism | c2560 iter-7 §1: the same probe cut wall by 113 us (1.7%) even though DVE's own stream was untouched. Pool stalls DVE through the semaphore interleave. |
| Expected gain | None directly. It sets the ceiling for every Pool-side lever. By proportion the c2560 figure suggests about 8 us (1.7% of 501 us) here. Treat that as unmeasured on c768. |
| Risk | None. It is not shipped. |
| Cheapest falsifier | One edit and one profile run, a few minutes. If wall drops by less than 3 us, close every Pool-side lever. |

Useful fact: 15 of the 168 Pool ADDs already run in 1.26 us instead of 2.16 us with identical operand shapes. They run while DVE is quiet, so the slowdown looks like SBUF port contention with DVE. I have not confirmed this. If that is true, the real lever is keeping Pool and DVE from hitting SBUF at the same time, not cheaper Pool opcodes.

### Candidate B: merge scan pairs across the two seq chunks (the only large lever left)

| field | value |
|---|---|
| Engine and member | `DVE0.bin` (96 scans deleted), `PE` (re-bank the B/C PSUM destinations), `Activation0.bin` (dA reorder), and a small `Pool0.bin` carry-address edit. |
| Change | Fuse `scan(chunk0,n)` and `scan(chunk1,n)` of the same channel tile into one `[512,2]` scan: `num_elem=[512,2]`, with a per-operand step equal to the distance between the two chunks' buffers. |
| Mechanism | c2560 iter-5 §4, with carry continuity proved bit-exact in its iter-1. c2560 iter-7 §0: moving a PSUM read earlier costs about 526 ns, while reordering non-PSUM slots is free. |
| Expected gain | 96 merges x about 126 ns = **12 us at most** (2.4%). c2560 iter-6 measured only about 80 ns per merge, so expect **about 7 us** (1.4%). |
| Risk | High. It needs window-private SBUF buffers for dBx, h and Cs, and a five-stream alias audit. The verifier would not catch a race. c2560 iter-6 built it, got it bit-exact, and measured it slower. |
| Cheapest falsifier | Merge 2 pairs only, check bit-exactness, and check that DVE busy time drops by 2 x 80 ns or more. If the gain per merge is under 60 ns, drop it. |

On c768 there are only 2 chunks per tile, so there is less to gain here than on c2560.

### Candidate C: shorten the head, where DVE is idle from 10.3 to 14.7 us

| field | value |
|---|---|
| Engine and member | The `SP0.json` queue descriptors and the first `PSEUDO_DMA_DIRECT2D` slots in `Pool0.bin`. |
| Change | The DMA table shows about 14 serial 1 KB copies on `qSyncIO0` finishing at 14.2 us, and the first compute starts at 14.67 us. Reorder or trim the loads so the first MULT's inputs land first. |
| Mechanism | Established by the profile only. I have not checked which DMA semaphore the first MULT actually waits on. |
| Expected gain | At most 3 to 4 us (under 1%). |
| Risk | Moderate. Reordering within a queue changes cumulative semaphore counts. The verifier would catch most mistakes because they would hang or fail. |
| Cheapest falsifier | In `SemaphoreUpdate.parquet`, find the semaphore the first DVE MULT waits on and the time it is satisfied. If it is satisfied within 1 us of 14.67, there is nothing to gain. |

### Candidate D: split the last y output DMA across two queues

| field | value |
|---|---|
| Engine and member | The final Pool DMA triggers at 508.0 and 509.5 us (tail). |
| Change | Split the final y group's output DMA by partitions, putting half on another queue or trigger. |
| Mechanism | The DMA flight time is about 2.8 us after the last trigger. |
| Expected gain | 1 us or less (0.2%). |
| Risk | Low, and the verifier catches mistakes. The queue semaphores differ per engine (c2560 iter-7 §3e says triggers cannot change engine). |
| Cheapest falsifier | Check whether the final DMA is bandwidth-bound (its duration scales with bytes) or latency-bound. If it is latency-bound, splitting gains nothing. |

### Not ranked: moving the remaining 24 DVE ADDs to Pool

This is closed by analogy. See the closed list below.

## Closed (do not redo)

All of these were measured on the c2560 sibling, not on c768. They are credible but are not re-measured on c768 unless noted.

- **Moving DVE work to Pool:** in c2560 iter-7, an ADD moved either way loses, and adding Pool work costs 0.68 to 0.89 us of wall per us. A real, bit-exact move of 142 `dt*x` ops to Pool made wall +449 us. On c768, iter-4's own commit already moved the 170 free ADDs, and the 24 left are the deadline-bound ones.
- **Moving work to ACT:** ACT cannot do tensor x tensor. ACT's idle time is also not usable, because its dA production runs in lockstep with DVE (c2560 iter-7 §3c). Carry COPYs on ACT gave a wrong result with `max_abs_err` 43.7.
- **Pool `PSEUDO_DMA_DIRECT2D`:** cannot change engine, and cannot be merged.
- **Pool access-pattern reshaping:** zero effect.
- **Operand swap and dtype changes:** swapping the PSUM operand order, and bf16 vs fp32 inputs, change nothing. DVE cost is about 203 ns per instruction plus 0.95 ns per element-op.
- **Removing ES slots and trimming waits:** zero effect (4000 inserted no-op ES slots changed nothing). Moving waits to remove DVE gaps conserves the gap. A wait-carrying ES inserted earlier costs about 280 ns.

## Unknown (would change the ranking)

1. **The Pool-removal ceiling on c768** (Candidate A). If it is over 8 us, a Pool-side lever such as stopping Pool and DVE hitting SBUF simultaneously jumps to first place.
2. **Whether Pool's 2.16 us vs 1.27 us ADD time is SBUF port contention with DVE.** To settle it, compare the 15 fast ADDs against the 153 slow ones for temporal overlap with DVE computes.
3. **Whether the compiler's PE bank depth of 4 can become 8 on c768.** This decides whether Candidate B's re-bank is safe.
4. **Whether the three de-aliased buffers in the last commit are race-free** under the five-stream audit. If one is not, the 501 us result is a lucky pass and the baseline for the next iteration is wrong.
