# Review of iter-4 (kernel.neff vs HEAD e833bbc; baseline 48.52 us)

Reference points: baseline 48.519 us, HEAD (iter-3) 40.03 us, working tree 38.35 us.
`notes/README.md` is still the empty template. The agent recorded nothing, so every claim below comes from the NEFF diff and my own profiling.

## 1. What changed

`neff_diff` against baseline shows 5 changed members. Against HEAD it shows 3: `Activation0.bin`, `DVE0.bin` and `SP0.bin`. `Pool0.bin` and every `.json` are identical to HEAD. I decoded the streams and diffed them instruction by instruction, because the raw slot diff is mostly an insertion shift.

- **Two more Vector CASTs moved to Activation.**
  - DVE slots 43 and 59 (`CAST`) became `EVENT_SEMAPHORE W S52>=1` and `EVENT_SEMAPHORE W S53>=1`. The DVE sem-6 count is unchanged (22 and 32 before and after).
  - Two new Activation `CAST`s were inserted. One waits `S6>=18` and updates S52; the other waits `S6>=28` and updates S53. They reuse the same src/dst, for example `0x180e800 -> 0x4300` and `0x180ea00 -> 0x4700`.
  - S52 and S53 are fresh semaphores. Each has one increment and one waiter.
  - Activation went from 80 to 82 slots.
- **DVE reorder.** `MEMSET(0x1804180)` now runs before `TT(S5>=58)`. The affected waits were re-pointed: DVE `CAST S6>=54` became `>=55`, Act `ACTIVATE S6>=55` became `>=54`, and DVE `TT S4>=35` became `>=36`. Waits on the swapped sem-6 counts match the swap.
- **Activation reorder.** Two ACTIVATEs (`0x1804180->0x1804b00` and `0x1804b00->0x1804400`) moved ahead of the bare `ACTIVATE 0x1005400->0x1004000`. This shifts sem-4 counts 39/40. SP `DMATRIGGER S4>=36` became `>=35`.
- **Final output store, SP slot 40.**
  - It was `PSEUDO_DMA_DIRECT2D W S4>=41` with `src var_offset 19456` (0x4c00).
  - It is now `W S6>=66` with `src var_offset 18192` (0x4710).
  - 0x4c00 was the result of the last Act `TENSOR_SCALAR` (slot 81). That op is `in_dtype=out_dtype=10` (the same 16-bit dtype), `op0=6`, `imm0=1.0`, so it is an identity multiply. The store now reads its input, the DVE `TT` slot 99 output, directly.
  - Act slot 81 is still in the stream but is now dead. Nobody waits on sem-4 count 41, so `semaphores.py` shows max_wait 40 against 41 increments.

**Consistency with the notes.** There is nothing to compare against, because the notes are empty. The HEAD commit message only describes the earlier edit. The decoded change is internally consistent and is mostly a re-pointing of waits.

The one behavioural change is the store bypassing the x1.0 copy. That is bit-exact for 16-bit-to-16-bit data. The verifier errors are identical to baseline to all printed digits (output_0 6.119e-05, output_1 2.864e-04).

## 2. Is the latency change real?

Yes. I re-ran both builds on the device with 7 captures each:

| build | median | min..max | spread |
|---|---|---|---|
| HEAD | 40.031 us | 39.993..40.406 | 1.0% |
| working tree | 38.349 us | 38.297..38.470 | 0.45% |

`profile_latest.json` has 38.417 (38.072..38.713, 1.67% spread), which agrees. The gain is 1.68 us, or 4.2%. That is about 4x the spread, so it is not noise.

## 3. Safety

- **Semaphore balance.** `semaphores.py` on the unpacked tree shows max_wait <= total_inc for every semaphore.
  - Sem 4: 41 increments, max wait 40.
  - Sem 6: 66 and 66.
  - S52 and S53: 1 and 1.
  - Compared with HEAD, only these changed: sem-4 max_wait 41->40 and sem-6 waiters 77->80. S52 and S53 are new.
  - No wait was weakened to a smaller threshold than its producer. The three re-pointed thresholds follow the swapped instruction order, which I checked by hand for the DVE pair and the Act pair.
- **Dead-op store bypass.** The SP store now waits `S6>=66`, the last DVE op. That is the producer of 0x4710, so it is correctly gated.
- **Not proven by the verifier.**
  - The two moved CASTs now wait only `S6>=18` and `S6>=28`. Before, they sat behind the DVE `TT W S5>=20` and `S5>=33`.
  - I did not independently prove that nothing still reads the CAST destinations (0x4300, 0x4700) when the new Act CAST writes them. The source side is covered by the DVE stand-in, which holds DVE until the CAST finishes.
  - The verifier runs one fixed input set, so a WAR slip would not show. Rerun the check with a second random seed if `bench/` allows it.
- **Error margin.** output_1 max_abs_err is 2.864e-04 against a limit of 3.5e-04, which is 82% of the limit. That is the same as baseline and was not moved by this iteration. Nothing is in the margin to spend, so any real numeric edit would fail.

## 4. Where the remaining time is

The kernel body is roughly 11.8 to 50.7 us on the trace clock. The trace also has a ~12 us runtime prologue on queue Q14.

| Engine | Busy | % of exec |
|---|---|---|
| Act | 20.6 us | 54% |
| DVE | 19.6 us | 51% |
| PE | 11.9 us | 31% |
| DMA | 9.8 us | 26% |
| Pool | 9.7 us | 25% |
| SP | 8.9 us | 23% |

No engine is above 55% busy, so the kernel is latency-bound, not throughput-bound. It is a chain of cross-engine hops.

- **Tail chain (measured).**
  - DVE `TT`97 (45.38-45.6) -> Act `ACTIVATE`78 (45.63-45.89).
  - -> SP `DMATRIGGER` slot 39 (WRITE 45.91). The source is 0x5100, produced by ACTIVATE 78, so the dependency is genuine.
  - -> SBUF->SBUF transpose DMA, about 1.34 us (Act slot 80 starts 47.25 on `S28>=32`).
  - -> Act `TS`80 (47.25-47.64) -> PE `MATMUL`145 (47.70-47.99) -> DVE `TT`99 (48.02-48.32).
  - -> SP `DMA_DIRECT2D` slot 40 (48.34, 668 ns to execute). The first data packet is at about 49.3. The last packet is at 50.1.
- **Spill/reload hops.** The queued DMA transposes (sems 24-39, qSPSpillReload0) are about 1.35-1.5 us each (SP37 WRITE 45.12 -> PE141 released 46.54) and sit on the critical path 3 times or more.
- **Mid phase (24-37 us).** PE is saturated here, at about 100% of each 2 us bin. This part is not reducible without changing the matmul count, which is an arithmetic change.
- **Prologue (kernel start to about 24 us).**
  - PE is idle for most of it.
  - Pool issues 5 DGE rows one after another (13.0, 13.8, 14.4, 15.1, 15.9; about 0.7 us each). SP is idle from 12.6 to 15.6.

## 5. Ranked candidates for iter-5

| # | Candidate | Engine / slots | Change | Mechanism | Gain / ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Replace the tail output stores (DGE) with queued transfers | SP slot 40 (`PSEUDO_DMA_DIRECT2D`, 64 KB) and Pool slot 20 (`PSEUDO_DMA_DIRECT2D`, rows [64,1] esz512). Maybe Pool 18. | Add a `dma` entry to `SP0.json` / `Pool0.json` with a descriptor whose `to` is the output variable and whose `from` is SB 0x4710. Replace the DGE with `PSEUDO_DMATRIGGER`, with the same wait. Re-balance the semaphore: the transfer signals the queue's `semaphore_set[id%16]`, so the current S17/S16 waiters, `Pool NOP W15`, and the final-drain waits must be re-pointed. | A queued trigger lowers to ~45 ns WRITEs (profile shows 43-59 ns each). A DGE costs 634-717 ns on the issuing engine. docs/09-dma.md 9.1-9.3 documents it as verified for SB->SB, but DRAM `to` variables are only shown for `from=input`. Unverified for an output variable. | Up to ~0.6 us (SP40 668 ns, Pool20 676 ns). Real gain could be about half, since the DMA ring latency stays. | Wrong variable name or offset gives wrong or missing output, which the verifier catches. A semaphore miscount hangs, which the first run shows. | Check profile timing: queued WRITE -> first packet latency against DGE start -> first packet (0.97 us measured at SP40). If it is under 0.4 us, proceed. If the DRAM-destination `dma` entry is rejected at load, it dies in one run. |
| 2 | Remove the dead Act `TS` slot 81 | Act slot 81 | Replace with an `EVENT_SEMAPHORE` stand-in that keeps sem-4 count 41. | docs/14 14.2: removing an unneeded instruction was verified. | About 0: Act drains at 48.69, before the DMA completes at about 50. | Very low. | None needed. Do it only together with other edits. Do not spend a measurement on it alone. |
| 3 | Spread the prologue DGE rows | Pool slots 7-11 (sems 9-13), SP idle 12.6-15.6 | Move Pool rows 9 and 10 to SP (the Activation and Sync engines accept DGE). Re-point the sem-9..13 DGE ownership. | Used in iter-2/3 for the DGE host moves. | Unknown, at most about 1 us. Row 11 (sem 13) is waited at 23.77 by SP20, so it is not critical. Consumers of rows 9 and 10 start 16.8 and 18.8, later than the rows complete, so something else gates them. | Low (moves only issue host). The verifier catches nothing wrong with timing, but a sem-owner mistake hangs. | Print `evt_wait_time` for DVE 13/14/20/21 and Act 7/8/11/12. If they are not waiting on S9..S12 at start, the candidate is dead. |
| 4 | Replace one DMA-transpose hop with a PE identity-matmul transpose | The last hop (sem 28 second round, SP trigger slot 39 -> Act slot 80) | Add an identity tile in SBUF, one PE `MATMUL` into PSUM and a copy back, in place of the SP trigger and DMA. | Exact for bf16/fp32 via identity (PE passes each element through once). Nothing in the notes or docs shows this done in a NEFF edit. | Up to about 0.9 us per hop (1.34 us DMA vs about 0.4 us PE+copy). Three or more hops on the critical path. | High. Needs a PSUM allocation and a new SBUF constant. Adds semaphore chains. The verifier would catch wrong values but not a PSUM bank conflict on other inputs. | First time one hop: add only the instructions as a standalone check. If the matmul/copy chain is not shorter than 0.9 us on the profile, stop. |

Candidates 1 and 3 are cheap to try. Candidate 4 is the only one that attacks the largest measured chunk. Do not start it until 1 and 3 are done.

## 6. Closed (do not redo)

- **SBUF->DRAM->SBUF staging** is fused into direct SBUF->SBUF. DMA bytes fell 377 KB -> 99 KB and spill/reload went to 0 (baseline profile vs now).
- **Pool/Sync DGE rows moved** off busy engines (iters 1-3). Pool 11.0 -> 9.7 us busy, DMA busy 17.9 -> 9.8 us.
- **Activation TS to DVE** and the pending-trigger hoists are already applied.
- **Dead x1.0 copy before the final store** (this iteration).
- **Throughput edits on Act or DVE** (more offloading). Act is at 54% and DVE at 51%, so there is no slack to harvest from the busy side.
- **Wait removal** is refuted for DVE and Pool (docs/14 14.2: wrong output, or an out-of-bounds runtime error).

## 7. Unknown

- **The ~12 us prologue on Q14.** Is any of it the kernel's own? The `total_exec_time` window is 38.35 us against a trace span of 52 us. I could not see exactly which window `profile.py` measures. Settle it by finding which timestamps `profile.py` subtracts, or by shifting the kernel's first instruction.
- **DGE vs queued issue cost for a DRAM destination.** Needs one measurement (candidate 1, step 1).
- **Whether Pool can take CAST or TS from Act.** Pool is 25% busy and idle 22-36 us. Act or DVE casts moved there would only help if they sit on the chain, and Pool per-op speed is unmeasured. Time one `CAST` on Pool to find out.
- **WAR safety of the two moved CASTs.** Needs a second input set through the verifier.
