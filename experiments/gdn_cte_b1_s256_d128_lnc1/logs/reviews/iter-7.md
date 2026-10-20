# Review of iter-7 (uncommitted `kernel.neff`, vs HEAD a7d7590 and vs `baseline/`)

Setup notes: `notes/` holds only the README, so there is no agent note to check the claims against. The only claims available are the commit messages. I unpacked HEAD, the working copy and the baseline to `/tmp/u_{head,cur,base}` and decoded them with `neff_doc/tools/isa.py` and `semaphores.py`. I re-measured the HEAD and working-copy NEFFs with `tools/profile.py`, then ran `--parquet` for per-instruction data.

## 1. What changed

The change touches 4 members: `Activation0`, `DVE0`, `Pool0` and `SP0`. It repeats the previous commit's trick: execute Activation `TENSOR_SCALAR_ARITH_OP`s on the idle Vector engine, with the same struct and `POINTER_IMMEDIATE` imm0.

- **Activation0** (121 slots, count unchanged; 5 slots differ)
  - Slots 9, 21, 23, 24 and 33 (`TENSOR_SCALAR MULT` by a pointer scalar) become `EVENT_SEMAPHORE` fences.
  - Slot 9's fence keeps its `sem4 +1` and now waits on new sem 53 instead of sem 9 >= 16. The other four are bare fences, because those ops had no waits and no signals of their own.
- **DVE0** (136 to 144 slots; the 130 "differing" slots are mostly index shifts)
  - Eight slots are inserted. Four are `TENSOR_SCALAR` clones of the moved ops:
    - one signals sem 53;
    - three write 0x1005880, 0x1005a80 and 0x1005c80 and each signal sem 70;
    - one more clone (slot 40 in the current stream) waits sem4 >= 14 and signals sem 71.
  - Three are `EVENT_SEMAPHORE`s with extended two-semaphore waits (sems 9+10, 10+11, 12+14, each >= 16) that carry the DGE-landing guards the Activation ops used to carry.
- **SP0**: one `EVENT_SEMAPHORE` inserted at slot 7, waiting sem 70 >= 3, in front of the staging DGE that reads 0x1005880 and the two after it.
- **Pool0**: one `EVENT_SEMAPHORE` inserted at slot 18, waiting sem 71 >= 1, in front of the DGE that reads 0x4d80.
- **DVE slots 57/73** (`CAST` to `EVENT_SEMAPHORE`) are index shifts and nothing more, not real edits (HEAD's own `DVE` diff shows the same pairs).

This matches the HEAD commit's description (a second batch of the same transformation). There is no commit message or note for this iteration, so "consistent with the notes" can only be checked structurally. Semaphore accounting is consistent:

- `semaphores.py` on the fixed path shows no "wait exceeds increments" flag.
- The sem 4 and sem 6 totals are preserved: sem 4 stays 61, sem 6 stays 93.
- New sems 53, 70 (3 incs) and 71 are each waited with the matching value.

## 2. Is the latency win real?

Yes. Three interleaved rounds of `profile.py --repeats 5` (HEAD / current):

| Round | HEAD median (us) | Current median (us) |
|---|---|---|
| 1 | 55.05 | 53.26 |
| 2 | 55.03 | 53.27 |
| 3 | 55.36 | 53.37 |

- The gain is about 1.8 us, or -3.3%.
- Within-run spread is 0.8-1.5%, and the two groups do not overlap (the worst current run, 53.8, is below the best HEAD run, 54.85).
- `profile_latest.json` reads 53.247 us (spread 1.22%), versus 67.525 us for the baseline. That is -21.1% cumulative.
- This is not noise. Per-iteration gains are now about 1-2 us, the same size as the spread, so a lone single-run A/B is no longer enough. Keep using interleaved rounds.

## 3. Safety beyond the verifier

Verifier: both outputs pass, with max_abs_err 6.09e-5 and 4.494e-4 (83% of the 5.4e-4 limit). I verified the baseline NEFF: its error is identical (4.494e-4, 83%), so the error has not moved. It is inherent, not a regression. Any edit that changes numerics, for example casts or accumulation order, has only about 17% headroom.

Synchronisation:

- No wait was removed or reduced on a live path. The producer's signal moved from sem 4 to a new sem, and each consumer found by address search was re-gated:
  - SP DGE reading 0x1005880/a80/c80 waits sem 70 >= 3.
  - Pool DGE reading 0x4d80 waits sem 71 >= 1.
  - The Activation fence waits sem 53 and still emits `sem4 +1`.
  - DVE source-landing guards use extended waits on sems 9/10/11/12/14 >= 16.
- Residual, not demonstrated, risk, listed in order of how concrete it is:
  1. The moved op that writes 0x4d80 has a second reader: Activation slot 77 (`ACTIVATE`, src 0x1004d80), which waits only sem4 >= 36. It is not gated on sem 71. It is safe today only because that slot runs about 15 us later, or through in-order DVE completion transitivity. It is not guarded by construction. Cheapest hardening: add a sem 71 >= 1 wait on Activation slot 77.
  2. The three DVE writes to 0x1005880/a80/c80 are read by PE, DVE and SP at much later points, each gated on counters that sit behind them. The same slack argument applies.
  3. Sems 14 and 20/23/58/62/65/67 now have 0 observed waiters in `semaphores.py`. This is expected for DGE-completion sems whose consumers are the engines' extended-wait fields, which the tool does not count (sem 14: the waiter moved into a DVE extended wait). It is not a bug, but note it when you re-balance.
- The verifier uses one fixed input set. Because these ordering changes are timing-slack, not data-dependent, a failure would show up as a rare wrong value, not a systematic one. Run verify 3+ times when changing anything similar.

## 4. Candidate edits for the next iteration (best first)

Where the time goes (profile of the current NEFF; the kernel body runs about 12 us to 63.5 us in profile time):

| Window | Activation busy | DVE busy | Notes |
|---|---|---|---|
| 12-16 us | 33% | 47% | Pool is 95% busy issuing 7 input DGEs back to back (13.16 to 18.45) |
| 16-23.5 us | 89% | 61% | Activation-bound |
| 23.5-29.3 us | 44% | 27% | PE idle 0% |
| 29.3-47 us | 62% | 71% | PE 100%+ (PE-bound recurrence) |
| 47-63.5 us | 65% | 53% | Chain of about 13 queued transposes |

A serial chain of tiny (about 0.2-0.4 us) ACT/DVE/Pool/PE ops runs from 16.0 to 25.7 us. It gates the queued 32 KB copy (block 5, triggered at 25.95), which in turn gates the PE chain that starts at 29.34.

| # | Candidate | Slots and change | Mechanism | Expected gain / ceiling | Risk and falsifier |
|---|---|---|---|---|---|
| 1 | Offload the remaining Activation `TENSOR_SCALAR_ARITH_OP`s in the 17-26 us chain to the Vector engine | Activation slots 14, 17, 25, 27, 30, 31, 34, 35, 36. Same struct as iter-7: clone into DVE with a fresh signal sem, leave a fence on Activation that keeps any `sem4 +1`, add an extended wait for the guarding DGE sem, and re-gate every reader. Do the chain ones (17, 27, 30, 35) first. | Same as the last two iterations: `TENSOR_SCALAR` is runnable on both engines with a pointer immediate. Established on device by iter-6 and iter-7 (-4.2% and -3.3%). | Ceiling is bounded by the Activation busy time in 16-23.5 us (89%, about 3.2 us of nine 0.33-0.39 us ops) and by the 9.7 us chain length. Realistic 0.5-1.0 us; the last batch of 5 gave 1.8 us. DVE has slack (27% idle in 23.5-29.3 us) but is 61% busy in 16-23.5 us. | Medium. Each consumer needs the new sem; iter-7 shows this is routine. Falsifier: move only slots 17, 27 and 35, then run three interleaved rounds. A gain below 0.3 us kills the rest. |
| 2 | `ACTIVATE COPY` with a pointer scale, as DVE `TENSOR_SCALAR MULT` | Activation slots 2, 7, 10 and 13 (profile pc 65/70/73/76, each 436-437 ns, each `sem4 +1`). Re-encode ACTIVATE (func COPY, `scale=[ptr]`, dst on PE-side addresses 0x4708/0x4780/0x5300/0x5700) as a DVE TENSOR_SCALAR MULT with imm0 = POINTER_IMMEDIATE, then re-gate PE (`LDWEIGHTS` and `MATMUL` waits on sem4) to the new DVE sem. | A COPY with a per-partition scale is an fp32 multiply; DVE `TENSOR_SCALAR` does it in about 285 ns, versus 437 ns on Activation. The ACTIVATE-to-TS re-encode is the new part and is not yet established on the device. | About 0.15 us per op, 4 ops on the chain, so 0.6 us at most (about 1.1%). Unknown until measured. | Medium to high, because it is a new struct. Check the output is bit-identical (the baseline error is already 83% of the limit, so there is little headroom). Falsifier: swap one op (slot 2 or 7), check it is bit-identical, and run an A/B. If the output differs, it is dead. |
| 3 | Take non-urgent staging DGE traffic out of the way of the critical 32 KB queued copy | Pool DGE rows for sems 62, 63 (profile 26.25 and 26.90 us) and Sync DGE rows for sems 60, 61 (26.77 and 27.42 us): add a guard wait on sem 29 >= 16 (copy complete) so they issue after the copy, or push them behind the copy trigger. Consumers are at Activation slots 46/47 (about 32 us). | Block 5's 16 data packets (2 KB each) start staggered from 26.8 to 28.6 us and take 0.23-0.49 us each, overlapping about 390 KB of DGE writes on the same DMA engines (Q2 24.9-26.6 and 27.6-29.0 us; Q1 27.5-28.7 us). Uncontended 16 KB transposes finish in 0.3-1.0 us. The contention is inferred from timing overlap, not proven. | Ceiling about 1.4 us at the copy (2.77 us trigger to done). The downstream chain is also gated by DGE 58/59 completing at about 29.0 us, so realistic is 0.5 us or less. Deferring 60-63 may instead stall Activation slots 46/47 at about 32 us. | Low for correctness (only issue times and an added wait). Perf risk is real. Falsifier: add the one sem 29 wait to the two Pool DGEs only, then run an A/B. If the copy's end time in the profile does not move earlier than 28.7 us, drop it. |
| 4 | Cut the 3.1 us first-input latency (12.13 DGE issue to 15.26 us DVE `CAST`) | First load: Sync DGE sem 8, bf16 to fp32 cast in the DMA with 2-byte source and 4-byte destination elements, about 272 tiny descriptors (Q2, 13.66-14.90 us). Change the pattern to contiguous bf16 and cast on DVE. | The DMA cost is per descriptor. It is the head of the whole dependency chain, so any saving is a straight shift. | At most about 0.8 us of DMA time, minus the cost of an added DVE cast op (about 0.2-0.3 us). Net 0.3-0.5 us at best. Unknown. | Medium to high: changes the layout of the first tensor and every consumer's dtype, which is a bigger edit than it looks. Falsifier: read the DGE descriptor in `Sync0.bin` slot 1 and confirm a contiguous layout is even possible. |

I have no honest fifth or sixth candidate. The remaining time is a chain of about 13 dependent queued transposes plus the 5-hop gate chain, bounded by DMA trigger-to-done latency (0.9-1.2 us each). Removing it would mean replacing DMA transposes with PE transposes. That adds instructions, PSUM and an identity matrix, which is a recompile-scale change and not a NEFF patch.

### Closed (do not redo)

- **Hoisting transposes 7-13 (SP slots 18-24) into the idle SP window at 19.8-23.5 us**: not worth doing. Those 7 triggers fire at 29.5-30.0 us only because of SP program order, and they do not read the copy's output. But they finish at 30.3-31.1 us and their first consumer (PE at pc 99) runs at 32.48 us, so they are never waited on. Ceiling is 0.
- **Dropping SP's `EVENT_SEMAPHORE` waits before the tail transpose triggers**: the one I checked (SP slot 44, waiting sem 30 + sem 44) is a real dependency. Block 22 reads 0x101cd00, which is the destination of the Pool DGE on sem 44. Do not remove them. I did not check the other waits in the tail chain.
- **Moving the Activation fences at slots 39/40 (waiting sems 58/59/57)**: Activation `CAST` at slot 41 waits on 56. The fences delay it from about 26.6 to 29.16 us, but nothing critical waits on that `CAST`. PE at 29.34 us is gated by the copy (sem 29), not by that `CAST`. Ceiling about 0.
- **Already done in earlier commits** (from the git log): fusing SBUF-DRAM-SBUF round trips; moving the DGEs hosted on Pool; per-block waits instead of 4-to-1 fences; the tail store split; first load on Sync. Cumulative 67.525 to 53.25 us.
- **`Pool` as a DGE host after an idle stretch** is about 500 ns slower than Sync at answering a semaphore (HEAD commit); keep trigger-critical DGEs on Sync or Activation.

### Unknown (would change the ranking)

- **Does the copy slow down because of DMA contention (candidate 3)?** Settle it by profiling with the two Pool DGEs deferred, or by dropping their row count in a throwaway build. If the copy's data-packet span falls from 1.9 us to 0.5 us, candidate 3 rises to the top.
- **Is Activation-to-DVE offload limited by DVE queueing (candidate 1)?** Compare the critical-chain end time (DVE `CAST` at 25.73 us, profile pc 104) after moving 2 ops against after moving 4. If it does not drop further, the chain is latency-bound and not engine-bound.
- **Does an `ACTIVATE COPY` re-encode to `TENSOR_SCALAR` give bit-identical output (candidate 2)?** Needs one op swapped and `verify.py` run.
- **Is there transitive protection for Activation slot 77's read of 0x4d80?** A single added wait on sem 71 settles this, and it costs nothing if the timing slack is real.
