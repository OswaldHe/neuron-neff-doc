# Review of iter-5 (kernel.neff vs HEAD 73236ba; baseline = unedited NEFF)

Method: `neff_diff.py` on baseline/HEAD vs working `kernel.neff`, plus my own decode of all 2952 changed slots (script at /tmp/r5/d.py), `semaphores.py` on freshly unpacked HEAD and new NEFFs (/tmp/r5x/{head,new}), and a fresh `tools/profile.py --repeats 1 --parquet` (/tmp/r5x/prof).
`notes/README.md` is still the empty template. The agent wrote nothing, so there is no claim to check the diff against and no refutations recorded for the next agent.

## 1. What changed

Only `sg00/Activation0.bin` changed. It has 14074 slots both before and after (no slot moved), and the other 36 members are byte-identical to HEAD. 2952 slots differ, in three families (two per site, three at a few odd sites):

| count | slot change | meaning |
|---|---|---|
| 984 | ACTIVATE -> EVENT_SEMAPHORE | the per-unit PSUM->SBUF `COPY` (func 128, scale 1.0, PSUM `0x2000800`-class -> SBUF `0x9ff8`-class) is deleted and the slot is reused as a nop that keeps the `SEM_INC_COMPLETE` on sem 4 |
| ~981 | `exp(-x)` ACTIVATE (func 7, scale -1) | its source changes from the SBUF copy to the PSUM tile, and the same-slot semaphore fields move (see below) |
| ~917+64+3 | `exp(+x)` ACTIVATE | the in-place `exp` on the SBUF copy becomes a PSUM->SBUF `exp` with `SEM_INC_COMPLETE` on sem 4 |

Example, first site (slots 13-16):
- Before: 13 = copy PSUM->SBUF (wait sem5>=6, inc sem4). 15 = exp(-x) SBUF->SBUF (wait sem4>=6, no inc). 16 = exp(x) in place (wait sem4>=6, inc sem4).
- After: 13 = exp(-x) reading PSUM directly (wait sem5>=6, no inc). 15 = exp(x) PSUM->SBUF (wait sem4>=5, inc sem4). 16 = EVENT_SEMAPHORE (wait sem4>=6, inc sem4).

So the iter-1 `exp(-x)` rewrite left a scaled PSUM->SBUF copy, and this iteration deletes it at 984 of the 1024 sites by letting both exps read PSUM. The decoded change is bit-exact in intent: the same fp32 values go through the same exps.
Without notes I cannot say why the remaining ~40 units were skipped (the odd-shaped 3+3 and 64-slot variants are part of the 984, so the leftover is ~40).

## 2. Is the latency change real?

Yes.
- `profile_latest.json`: median 4777.85 us, samples 4776.6-4778.3, spread 0.035%.
- My own re-run gave 4777.17 us.
- HEAD was 4856.3, so the delta is -78.5 us (-1.62%). That is about 45x the run-to-run spread.
- Against the 7154.9 baseline (spread 0.021%) the total is -33.2%.
- ACT busy dropped 3672 -> ~3345 us (profile json). The wall gain is ~19% of the ACT busy removed, which matches the earlier finding that ACT busy converts poorly to wall time.

The loop is not chasing noise this time. The usable resolution of this profiler is about 1-2 us.

## 3. Safety beyond the verifier

- **Semaphore totals are conserved.** In the HEAD and new `semaphores.py` runs the only diff is the sem-4 source split: ACTIVATE 7263 -> 6296, EVENT_SEMAPHORE +967. Total sem-4 increments are 7360 in both, and no sem 4 `max_wait > total_inc`. The 32 "wait exceeds increments" lines (sems 40-71) are identical before and after.
- **No wait is released earlier.** I mapped every sem-4 wait threshold T to the slot of the T-th ACT increment, before and after. Across 9698 waiters (ACT 4241, DVE 2746, PE 2465, SP 226, Pool 20) none is released earlier, 7000+ are unchanged, and 1460 are released later (ACT self-waits 990, PE 408, DVE 34, SP 27, Pool 1). This is the property that matters: PE waits on sem 4 to know the PSUM bank is free, and the PSUM tile is now read until slot 15 instead of slot 13, so PE's wait moved with it. This is why it is race-safe by construction rather than by luck.
- **Margins.** The verifier ran only 1 run and 1 input set. Max abs error is 5.234e-7 of a 6.2e-7 limit (84%) for output_0 and 3.33e-7 (54%) for output_1. The baseline was 5.122e-7 (83%) and 3.17e-7 (51%), so the error already moved up with earlier iterations. I believe this iteration is bit-identical to HEAD but did not diff the output tensors (the iter-4 commit message claims bit-identical for its own edit). The next numerics-touching edit has only ~16% headroom, so it must be checked bit-for-bit against HEAD output rather than against the limit.
- **Not verified:** the 3 non-standard sites (activation_func 128->7 with different src/dst and `events_semaphore_value` 5150->5149) are covered by the sem analysis above, but not separately by data-flow reasoning.

## 4. Ranked candidates for iter-6

Profile facts the ranking rests on (`/tmp/r5x/prof`, span 4791 us):
- ACT busy 3935 us (82%), of which ACTIVATE 9973 instr = 3604 us. Within that, `COPY(scale/bias)` in-place or SBUF->SBUF is 6865 x ~430 ns, about 3.0 ms.
- DVE busy 2824, of which STREAM_SHUFFLE 6012 x 147 ns = 885 us and COPY-from-PSUM 959 x 214 ns = 205 us.
- Pool does DMA_DIRECT2D only (3169 x 655 ns) and sits idle ~2.9 ms, waiting on Vector.
- ACT stalls 1045 us waiting on Vector. PE stalls 1842 us waiting on Scalar and 769 us on Vector.
- A greedy backward critical-path walk was inconclusive: I had to filter out LDWEIGHTS, so it was dominated by the in-order PE chain. Do not treat it as evidence that PE is critical.

### Candidate A (best): fuse "DVE PSUM->SBUF copy, then ACT in-place scale" into one DVE multiply, ACT slot becomes a nop
- **Where:** DVE `COPY` with PSUM source (959 profiled; 932 are followed by an in-place ACT scale on the copy's destination) and the matching ACT `COPY(scale/bias)` slot with `src==dst`.
  Example: DVE copy PSUM `0x2002800` -> SBUF `0x5a18` [64], then ACT in-place `COPY` `0x5a18` with scale `[0x5908]`.
- **Change:**
  - DVE slot becomes a TENSOR_TENSOR MULTIPLY: src0 = PSUM tile, src1 = the ACT scale cell with free-dim stride 0, dst = the copy's destination. Build it from an existing DVE `TENSOR_TENSOR MULTIPLY` slot (they exist) and keep the DVE slot's wait and update fields.
  - The ACT slot becomes an EVENT_SEMAPHORE that keeps the ACT slot's own wait (sem6 >= N) and its update (sem4 inc, if any).
  - No threshold anywhere changes, so the sem-conservation argument from iter 4 and iter 5 applies unchanged. PSUM is freed by the same DVE sem-6 increment as before, so there is no new PE race.
- **Mechanism:** same placeholder discipline as iter 4 (EVENT_SEMAPHORE keeping the increment) and this iteration. Evidence that DVE TENSOR_TENSOR reads PSUM and takes a stride-0 broadcast operand: 2064 existing TT instructions have a PSUM src1, and others use `[1,1,0]` patterns.
- **Expected gain:** removes ~932 ACT ops x 430 ns = ~400 us of ACT busy and one DVE->ACT->DVE hop per unit. DVE changes from a copy to a multiply at roughly the same cost (~215 vs ~225 ns). At the measured conversion rates (1.6% in iter 2, ~19% here) expect -10 to -80 us wall. **Ceiling: ~-80 us** (ACT busy removed x 19%). Unknown if the hop removal beats the busy-based estimate.
- **Risk:** medium. Numerics are the main one: ACT scale multiply vs DVE multiply could differ on denormals or flush-to-zero, and there is only 16% error headroom. The synchronisation is safe by construction. The verifier would catch a numerics change if it is large, but may not catch a rare denormal one, so compare outputs bit-for-bit against HEAD output.
- **Cheapest falsifier:** patch one site, run, and diff the output tensors bit-for-bit against HEAD output. A single mismatching element kills it. Then patch all 932 sites and read the median (only worth ~1 minute of profiling).

### Candidate B: finish the iter-3 shuffle dedupe at the remaining ~479 units
- **Where:** DVE STREAM_SHUFFLE count is 6012 = 4096 + 479 x 4, so the second per-unit exp_cs_last broadcast (4 shuffles of 147 ns) still exists at 479 units. Iter 3 replaced it with a DVE COPY at only 545 of 1024 sites and left no note on why the others were skipped.
- **Change:** same replacement (4 STREAM_SHUFFLE -> 1 DVE COPY of the first broadcast, other 3 slots -> EVENT_SEMAPHORE keeping their updates) at the remaining sites if the first broadcast's destination is still live and unmodified there.
- **Mechanism:** the iter-3 edit, established in commit 5b5d3e5 (-0.42%, DVE busy 2701 -> 2440).
- **Expected gain:** ceiling is the DVE time removed: 479 x (4 x 147 - 190) = ~190 us DVE busy. At iter-3's measured conversion (261 us busy -> 21 us wall) expect ~-10 to -20 us. **Ceiling ~ -20 us** (unmeasured for this half).
- **Risk:** low-medium. The reason these sites were skipped is unknown and may be a real hazard (destination overwritten, or a different address layout). The verifier would only catch it if the data differs on the fixed input.
- **Cheapest falsifier:** read why the first 545 qualified and the others did not (compare the addresses of the two broadcasts at one skipped site). If the second broadcast's content differs from the first, it dies in a minute.

### Candidate C: use the idle Pool engine for ACT scale work, in the 1920 dead Pool slots from iter 4
- **Where:** Pool stream placeholders (EVENT_SEMAPHORE nops that replaced the state_sb DMA triggers; Pool busy 2.07 ms of 4.78 ms, waiting on Vector 2.3 ms). The 1873 in-place ACT scale ops that do not follow a DVE copy are the work to move.
- **Change:** turn a Pool nop into a TENSOR_SCALAR / TENSOR_TENSOR multiply that does the ACT op's work, with the same wait and update fields transferred, and turn the ACT slot into an EVENT_SEMAPHORE nop.
- **Mechanism:** Pool can execute elementwise ops. Not established on this device: no Pool compute op other than one COPY (589 ns) appears in this kernel.
- **Expected gain:** unknown. ACT busy would drop ~430 ns per moved op, but the Pool slot position was fixed by the old DMA schedule and may not sit at a point where the operands are ready. A late Pool slot would delay consumers.
- **Risk:** high. Pool increments are on the DGE semaphores 8-23 (+16), not sem 4, so the ACT consumer's sem-4 count must move, which is a threshold re-balance. A RAW/WAR mistake is invisible to a fixed-input verifier.
- **Cheapest falsifier:** microbenchmark one Pool multiply of 128 x 64 fp32 in a free Pool slot and read its duration in the profile. If it is not clearly under ~430 ns, or the operand-readiness is not already guaranteed by the slot's existing wait, drop it.

### Candidate D: drop the 967 EVENT_SEMAPHORE placeholders on ACT (and re-balance sem 4)
- **Where:** the 967 `Activation:EVENT_SEMAPHORE` slots this iteration created (slot 16 at each site).
- **Change:** delete the slot and decrement every sem-4 wait threshold by the number of removed increments before it (the mapping script in this review can compute it).
- **Mechanism:** none documented for removing slots. Deleting a slot moves all following PCs, which is not shown to be safe in the docs I have. Leaving the slot as an unconditional nop does not help: an EVENT_SEMAPHORE costs ~40 ns regardless.
- **Expected gain:** **ceiling 967 x 40 ns = ~39 us ACT busy**, at the ~19% conversion rate about -7 us. Not worth the risk.
- **Falsifier:** check first whether any engine tolerates slot deletion. That is a bigger question than the gain.

### Candidate E (low priority): shave 64-element ACT ops by checking what dominates their ~430 ns
- 64-element ACT ops take ~430 ns, while 1-element ops take 150-200 ns. That implies ~4 ns per element, which is far slower than a nominal one element per cycle.
- If this is an addressing/stride effect rather than fp32 throughput, a field change in the pattern (e.g. a layout detail in `src_mem_pattern` or `num_active_channels`) might lower it, but I have no evidence for such a field. This is a measurement task, not an edit.
- **Cheapest check:** time a single test activation at 16, 32, 64, 128 elements by editing `[64,1,1]` and read the profile. If time scales linearly with 4 ns/elem there is no free lunch.

## Closed (do not redo)

- **Table reloads / reciprocal:** 2040 of 2046 ACT_TABLE_LOAD removed in iter 1 (7154.9 -> 5130 us, -28.3%). Only 6 remain (6 x ~1.28 us).
- **Adjacent identical ACT pairs merged (iter 2):** 303 pairs, ACT busy -97 us gave wall -1.6 us. Merging more ACT ops only buys wall at ~2% of the busy removed.
- **Redundant second exp_cs_last broadcast (iter 3):** -0.42% at 545 sites (the rest are Candidate B).
- **state_sb HBM round trip (iter 4):** 1920 DMAs deleted, 5107.0 -> 4856.3 (-4.91%). Nothing left of that DMA family on Pool; the Pool DMA_DIRECT2D that remain are the HBM in/out.
- **This iteration's PSUM->SBUF copy before the two exps (984 sites):** -78.5 us. The only PSUM->SBUF copy left on ACT is `Scalar COPY psum` 98 x 213 ns = 21 us total, not worth a pass.

## Unknown (would change the ranking)

- **What is the real critical path?** My walk was inconclusive. Measure it properly by including LDWEIGHTS and taking the latest predecessor per instruction, then attribute time by engine per unit. If PE is on the path, Candidate A buys nothing and PE-side work moves up; if it is the ACT<->DVE ping-pong, Candidate A moves to the top.
- **ACT busy -> wall conversion factor.** Observed 1.6% (iter 2) and ~19% (iter 5). If more work like Candidate A converts at 19%, ACT work is still worth removing. A single A/B (patch 100 sites of Candidate A, read the slope) measures it.
- **Why 479 units were skipped in iter 3** (Candidate B precondition).
- **Whether DVE vs ACT multiply is bit-identical on denormals** (Candidate A). Settled by one bit-exact output diff.
- **Does a Pool multiply on 128 x 64 fp32 beat ACT's ~430 ns?** (Candidate C)
- **Verifier coverage:** runs 1 input set. For any change that touches ordering, repeat verify several times (`--runs`) before accepting.

## Bottom line

The accepted edit is real (-78.5 us, -1.62%, spread 0.035%) and synchronisation-safe by construction (sem-4 totals conserved, no waiter released earlier). The remaining levers are small. Candidate A has the best gain per risk, with a ceiling of about -80 us (~1.7%) and Candidate B about -20 us; everything else is ≤1% or unproven. If A and B both land at the low end, the kernel is close to what NEFF-level edits can reach without restructuring the scheduling across engines.
