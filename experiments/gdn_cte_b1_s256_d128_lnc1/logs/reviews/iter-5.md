# Review of iter-5 (kernel.neff uncommitted on top of ddb2b42)

## 1. What changed
`notes/README.md` is still the empty template, so there is no claim to check against. Diff is against HEAD (ddb2b42, 59.4 us); vs. baseline the earlier commits (SBUF->SBUF fusion, DGE spreading) are included.
Changes vs HEAD (3 members, no other member differs):
- **Activation0**: slot 88 PSEUDO_DMA_DIRECT2D (0x101ce00 staging DGE, sem 43-ish) turned into an EVENT_SEMAPHORE nop; slots 90-93 rotated (a nop, TENSOR_SCALAR, nop, ACTIVATE moved by one); a DGE re-appended as new slot 120 (end of stream, 0x4b00->0x101500 region, new tail slot). Net: one DGE moved from mid-stream to the stream tail, plus 1 new slot (120 -> 121 slots).
- **Pool0**: slots 21/22 (two var9 reload DGEs) swapped and their guard semaphores re-pointed (sem 44<->66, wait 42<->63); slots 24-26 wait-only EVENT_SEMAPHOREs reduced to pure nops; slots 27/28 (DGE vs AFFINE_SELECT) swapped; slot 35 wait value 16->32 (now waits on both DGEs of sem 21). Slot 1 became a nop and the DGE moved to SP.
- **SP0**: first ~9 slots re-ordered (a DGE moved from slot 6 to slot 1, the PSEUDO_DMATRIGGER block ids shifted 1..5 -> 0..4 and wait values/idx re-pointed); 58 slots differ mostly as a shift.
So: pure scheduling -- DGEs hoisted earlier in the in-order issue stream of SP/Pool, sem waits regrouped. No data-path field (addresses, dtypes, shapes) changed except swaps of identical-shape pairs.

## 2. Is the speedup real?
Yes. I measured both NEFFs back to back with `tools/profile.py`:
- HEAD (ddb2b42): 59.475 / 58.845 / 59.523 / 59.418 -> median **59.45 us** (spread 1.1%)
- this iteration: 57.935 / 57.502 / 57.446 / 57.524 / 57.489 -> median **57.50 us** (spread 0.85%); `profile_latest.json` itself says 57.90 (spread 1.6%).
Delta -1.95 us (-3.3%) is ~2x the within-run spread and reproduced across two separate profile runs. Real, not noise. vs baseline 67.525: -14.8%.
Caveat: the loop is now at the point where candidates of <=0.7 us are ~1x spread; repeat >=5 repeats and compare against a same-session HEAD run.

## 3. Safety
- Verify passes: max_abs_err 6.09e-5 = 11% of the 5.4e-4 limit; no NaN/Inf; one input set only.
- `semaphores.py` on the unpacked NEFF: no "wait exceeds increments" on either HEAD or this one. Waiters per sem otherwise unchanged except: **sem 22 and sem 44 now have 0 waiters** (each still gets +16 from a DGE: SP-sem22, Pool-sem44; before, HEAD had 1 waiter each); sem 21 now collects 2 DGEs (Activation + SP) with a single wait for 32; sem 8 moved from dge:Pool to dge:SP; sem 16 (single Pool nop) went away.
- Concern: the Pool DGE writing 0x101cd00 (sem 44) has **no consumer found by address** (no instruction in any member references 0x101cd00 as a source; PE[134] reads 0x101ccfe with 128 elems, which spans the end of the 0x101cc00 block). If that is an overlap read, the only ordering guarantee is now the sem 43 wait on the Activation DGE, not on sem 44. It passes today because the Pool DGE is issued much earlier; a slower DMA queue could race. Not caught by a single-input verify. Check: stress with a second random input and with an artificial delay (insert a 2 us dummy before Pool DGE 22) -- if output changes, the wait is needed.
- Error is nowhere near the limit; dtype fields unchanged (FP32<->BF16 swaps are between two identical-pair DGEs and moved together).

## 4. Candidates for next iteration (best first)
Evidence base: per-instruction table of this NEFF (prof/Instruction.parquet). Facts used: DGE descriptor-issue instructions (PSEUDO_DMA_DIRECT2D) cost 600-930 ns of the *issuing engine* each (Sync 11x avg 675 ns = 7.4 us, Pool 15x avg 678 = 10.2 us, Act 4x 651 = 2.6 us; 30 DGEs ~20 us total). Tensor engine idles 10.0 us at t=21.5->31.5 us (pc 82) waiting for the DGE round trip; Pool idles 17.5 us at 29.6->47.1; Act busy 58.8%, DVE 49.5%, Pool 27%, SP 20.6%. Preamble ~12 us (TENSOR_LOAD / semaphore setup, rows 0-12 us) is runtime-inserted, treat as fixed.

| # | Edit | Mechanism | Ceiling | Risk | Cheapest falsification |
|---|------|-----------|---------|------|------------------------|
| 1 | **Spread the 3 final output DGEs** (Sync pc128 @66.99, Act pc183 @67.47, Sync pc129 @67.69 (933 ns)): move Sync pc129 (and ideally pc128) to Pool0 slot after 99 -- Pool is idle at the tail (waits until 69.9) | DGE queue follows its semaphore, not the issuing engine (ddb2b42 commit, earlier DGE moves); issue is serialised per engine | <=0.7 us (the 2nd Sync DGE starts 0.7 us after the first) ; bound: tail from last compute (67.4) to exit (70.4) = 3 us of which DMA latency stays | low; sem wait values must follow; verifier catches wrong data not races | profile before/after; if t(last DMA end) doesn't move >0.3 us, drop |
| 2 | **Re-balance the 28.2-30.1 us DGE burst**: Sync pc74-76 and Pool pc76-77 are 5 DGEs serial on 2 engines (~600 ns each). Put one on Act-or-DVE-free queue / SP-Pool 3-way split, or hoist them to the slot just after the last producer ACT (27.9) | same as #1; critical path: Scalar chain ends 27.9 -> DGEs 28.2-30.1 -> DMA latency -> Scalar CAST pc104 @31.3 -> PE pc83 @31.49 | <=0.7 us (5 serial -> 3 per-engine); bound = the 10 us PE gap | low-medium | profile; check CAST pc104 start moves earlier |
| 3 | **Split the final output store** so the first half issues as soon as ready (Act pc183/Sync pc128 cover the last tile; split by partition range) | latency of the last store is exposed: last compute 67.4 -> exit 70.4 | <=1 us, unknown (not measured how much of 3 us is DMA flight) | medium: extra DGE costs 650 ns on an engine; verifier catches wrong data | check DmaPacket timestamps for the last output queue: if flight < 1 us, drop |
| 4 | **Offload ~25% of Activation TENSOR_SCALAR** (33 ops, 11.8 us, ~360 ns each) to Pool (idle 73% of the time) | Pool/Act both implement TENSOR_SCALAR_ARITH_OP; Act chain 20.4-27.9 us is contiguous | <=1.5 us (chain length / 4); real figure depends on Pool cycle cost for a [128,x] op, unmeasured | high: new sem hops each way (+~100-200 ns), instruction encoding differs between engines; races not caught by verify | microbench one op on Pool via notes in neff_doc/experiments; if Pool op > 450 ns drop |
| 5 | Merge adjacent same-stride DGEs (SP slots 7-9, src stride 0x200, dst partition step 0x400000) into one | DIRECT2D has 2 dims, already used | unknown; 600-ns fixed issue cost suggests ~0.6 us per merged pair | high: needs a 3rd dim; probably impossible | check ISA struct: if both step fields already used, dead in 5 minutes |

Realistic total of #1-#3: 1-2 us (~2-3%), comparable to a 1-2x measurement spread -- each must be confirmed with 5+ repeats against a same-session baseline. The remaining time is mostly a dependency chain (Act->DVE->PE recurrence + DMA flight) plus a fixed ~12 us preamble, so I would not expect more than ~5% further.

**Closed** (from the git log; notes are empty):
- SBUF->DRAM->SBUF staging round trips: fused (86e9bd9, 67.5->61.9). Spill save/reload is now 0 bytes in the profile; nothing left there.
- Moving var9 reload DGEs off SP to Pool (9d8415c, -0.9 us), staging DGE queue spreading and per-block waits (ddb2b42, -1.7 us): done; do not redo.
- Hoisting the DGE issue ahead of its triggering wait (this iteration, -1.95 us): done.

**Unknown** (measure before spending effort):
- Real consumer of Pool DGE sem 44 / SP DGE sem 22 (zero waiters now): a second-input + delay-injected run settles the race question.
- DMA flight time vs issue time of the last output store: `DmaPacket.parquet` timestamps.
- Cost of TENSOR_SCALAR on Pool vs Act: one-instruction microbenchmark.
- Whether the 12 us preamble depends on anything in the NEFF (it is TENSOR_LOAD/EVENT_SEMAPHORE with a 6.6 us PE wait pc54): check whether those semaphores are generated by the loader, so far assumed fixed.
