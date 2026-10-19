# Review: iter-2 (kernel.neff vs HEAD d6d348b and vs baseline)

Tools used: `neff_diff.py` against both baseline and HEAD (the previous accepted iteration), `semaphores.py` on unpacked copies of both, a fresh `tools/profile.py --repeats 1 --parquet`, and the Instruction and DmaPacketAggregated tables.

## 1. What changed

`notes/` contains only the template README. Nothing documents this iteration, and the HEAD commit message describes iteration 1 only. So I could not check the change against any notes. What follows is reconstructed from the NEFF.

Relative to HEAD, 5 members changed:

| member | change |
|---|---|
| `def.json` | new DMA queue `qSPSpillReload1` (owner sp, 16 queues, semaphores 72-87) |
| `SP0.json` | +192 `fused-varN-*` transfers on `qSPSpillReload1`. Their ops are `copy` or `cast` (fp32 to bf16). The sources and destinations are SBUF addresses taken from the old save and reload descriptors. The original 669 `qSPSpillReload0` entries are byte-identical to HEAD. |
| `SP0.bin` | 1315 to 1507 slots. 12 new queue-1 `PSEUDO_DMATRIGGER`s appear per semaphore group (192 in all). The old SP-issued `PSEUDO_DMA_DIRECT2D` reloads (sems 56-71, 160 of them) became `EVENT_SEMAPHORE` nops. Each nop waits on the fused transfer's completion semaphore (`events_extended_wait` on sem 72-87, value 16k) and then adds 16 to the old semaphore, so existing consumers are unchanged. |
| `Activation0.bin` | 224 `PSEUDO_DMA_DIRECT2D` (SBUF to DRAM saves, sems 40-55) became `EVENT_SEMAPHORE`. The producer wait is kept (wait fields moved to the extended-wait position) and the same +16 on the same semaphore is kept. No data moves any more. These now act as a "producer ready" signal. |
| `Pool0.bin` | 32 `DMA_DIRECT2D` reloads (sems 8-21) became nops that wait on the fused-transfer semaphore. |

The mechanism is iteration 1's save→reload fusion, generalised: DRAM spill round-trips become SBUF→SBUF transfers on an SP-triggered queue. A semaphore-bridging nop is left behind so downstream waits stay intact. The profile agrees: `spill_save_bytes` and `spill_reload_bytes` go to 0, and `hbm_read_bytes` and `hbm_write_bytes` go from 12 MB to 3.1 MB, which is just the kernel's I/O.

## 2. Is the latency change real?

Yes.

- `profile_latest.json`: median 1208.0 us, samples 1201.9 to 1209.3, spread 0.61% (about 7 us).
- My independent run: 1205.3 us.
- The previous accepted iteration measured about 1312 to 1313 us. The gain over it is about 105 us (-8%), roughly 15 times the spread. It is not noise.
- Against baseline (1351.4 us), the total is -10.6%.
- Engine busy times fall where expected: Act 627 to 520 us, SP 131 to 52 us, DMA 420 to 340 us, Pool 255 to 233 us. PE and DVE are unchanged.
- `verify_latest.json` passes. max_abs_err is 1.097e-4 and 9.081e-4 (limit 1.1e-3), the same figures as the baseline verify, so the outputs look bit-identical. Output_1 sits at 83% of the limit, but it was already there in the baseline and did not move.

## 3. Safety beyond the verifier

- **`semaphores.py` has a blind spot.** It does not parse `events_extended_wait_*` on `EVENT_SEMAPHORE`. It reports sems 72-87 as "0 waiters". They do have 192 waiters, spread over 160 SP nops and 32 Pool nops. I decoded these by hand. Every nop has a wait ≥ 16k, and sems 72-87 each receive 192 increments (12 transfers × 16 queues). Nothing is waited on beyond what is incremented.
- **RAW ordering looks correct.**
  - Act nop = producer ready.
  - The fused trigger on SP waits on that nop's semaphore (idx 41, 42 and so on).
  - The completion nop on SP or Pool waits on the fused semaphore.
  - Consumers still wait on the old semaphores, and those only fire after the data has landed.
- **A WAR hazard is open, and the verifier would not see it.**
  - Before, the Act save's completion on sems 40-55 meant "the SBUF source has been read".
  - Now the Act nop fires at producer-ready, before the fused DMA reads the source.
  - Waiters on sems 40-55 that are not the SP fused triggers: 3 `Activation CAST`, 4 `DVE COPY`, 2 `DVE EVENT_SEMAPHORE`, 128 `Activation EVENT_SEMAPHORE`. Some of these may be buffer-reuse gates, so a later instruction could overwrite the source before the DMA reads it.
  - Fused-transfer start-to-end times are far from instant: 17 to 24 us for casts, and the average grows to about 150 us late in the run. Read timing is therefore not guaranteed to be tight.
  - Outputs are bit-identical on one fixed input set, so this is a latent race, not a demonstrated one.
- **Fix before building on this:** retarget those 9 non-SP waiters, or add a completion wait, so they wait on the fused semaphore or the SP nop's semaphore. Then run verify several times (`--runs`) on different random inputs.
- Verify ran once (`"runs": 1`).

## 4. Candidates for next iteration (ranked)

The profile gives these facts:

- No engine is saturated. In 100 us windows the busy fractions are PE about 62%, DVE about 49%, Act about 48%, Pool 15% and falling, SP 4%.
- PE has 192 idle gaps longer than 1 us, totalling 548 us (45% of the run). 156 of them end within 0.6 us of a `qSyncSpillReload0` completion, and 16 end at a `qSyncSpillReload1` completion. The latency is a dependency and handoff chain, not throughput.
- For the PE-gating reload0 transfers, the DMA starts on average 2.07 us after PE has gone idle. The DMA itself lasts 0.49 us. So most of each gap is the upstream producer, and the DMA hop is only about 0.5 us plus semaphore wake-up.

| # | Candidate | Engine / member / slots | Change | Mechanism and where established | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Close the WAR hole (correctness; do first, expected cost about 0)** | Act0.bin (3 `CAST`, 128 nops), DVE0.bin (4 `COPY`, 2 nops) waiters on sem 40-55 | Point each of these waits at the SP completion nop's semaphore (56-71) or at 72-87 with the matching cumulative value, instead of the early Act-nop semaphore | Extended-wait semantics as already used by the nops in this iteration | Zero gain. It may cost a few us; if the loss is large it kills this iteration's approach. | Low. Verifier cannot catch the race. | Run `tools/profile.py` and verify 5 times. A loss of more than 10 us means the early release was load-bearing. |
| 2 | **Shorten the hop in front of PE: reload0 trigger waits** | SP0.bin triggers for `qSPSpillReload0` that gate PE (about 156) | Compare each trigger's wait with the latest instruction that really produces its data. Where it waits on extra or late semaphores (for example sem 28 or the earlier reload chain), loosen it to the true producer's semaphore only. | Same trigger retarget as iteration 1 (`d6d348b`), which gave -2.9% | PE-gating reload0 DMA time sums to about 70 us (144 transfers × 0.49 us). Real ceiling is probably 20 to 40 us. | Medium. This is a sync loosening and the verifier will not detect a race. | For 20 gaps, measure trigger start minus producer end. If it is under 0.2 us the waits are already tight and the candidate is dead. |
| 3 | **Turn the 32 cast transfers into plain copies, with the bf16 cast moved to an existing Act/DVE op** | SP0.json `fused-var91-*`, the cast descriptors (about 20 us each) | Replace op `cast` with `copy` on the DMA, and have the consumer read fp32 or use an existing cast instruction | The cast transfers take 19 to 24 us, against 0.6 us for plain copies | Up to 16 PE gaps (3 to 8 us each), about 45 us at most | High. It changes numerics, which is why it is not simple, and needs the consumer layout to change. | Look at `fused-var91` consumers. If none is on a PE gap, drop it. |
| 4 | **Spread the fused queue over a second queue or ring** | `def.json` `qSPSpillReload1`, `SP0.bin` triggers | Split the 192 fused transfers across two queues so a late-ready trigger doesn't block unrelated ones (SP triggers in program order) | Queue and trigger layout in `def.json`. It has not been tried; the DMA-ring FIFO order is an assumption. | Small. Only 16 of 192 PE gaps end at a reload1 completion. At most about 45 us, probably under 15. | Medium: new queue means a new semaphore range. | Look at head-of-line blocking: is any fused trigger delayed past its inputs because an earlier SP trigger waits? If SP trigger wait time is near zero, dead. |
| 5 | **LDWEIGHTS reuse** | PE0.bin (1878 `LDWEIGHTS`, 208 us, 111 ns each) | Drop an `LDWEIGHTS` when the previous one loaded identical weights | Never measured here | Unknown. PE is only 62% busy, so most of this does not shorten the critical path. | High. A wrong match corrupts output. | Count adjacent identical-operand `LDWEIGHTS`. If there are no repeats, dead. |

If candidate 1 fails (large loss), the iteration's gains partly come from the race, and it should be treated as at risk of being rejected later. After candidate 1 the remaining NEFF-level headroom is small and speculative. This is mostly latency-chained across engines, and most of the 548 us of PE idle time is upstream compute, not DMA.

## Closed (do not redo)

- **DRAM spill saves and reloads on Act, SP and Pool.** Already fused or nopped. Spill bytes are 0. The HBM traffic left equals the kernel's real input and output.
- **Pool `DMA_DIRECT2D` (288 left, 202 us).** These are the kernel's real I/O (about 3.1 MB in and out), not spills. Removing them changes data.
- **"Engine X is saturated."** No engine is above 62% (PE) in any 100 us window.
- **Iteration 1's first 32 fusions.** Done (`qSPSpillReload0` entries are unchanged since HEAD).

## Unknown

- **Which instruction chain bounds the run.** A dependency walk on the profile would settle it: for each PE gap, find the last-finishing predecessor. This would rank candidates 2 to 4 properly.
- **Whether the WAR race can fire.** Run verify with different random inputs and the profile, with fused-trigger delay injected (a deliberate delay on the SP trigger, such as an extra wait), and see if output changes.
- **Whether `semaphores.py` should be fixed** so it counts `events_extended_wait_*`. Until then it gives wrong "0 waiter" results for these nops.
