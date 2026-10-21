# Review of iter-7 (HEAD 51c6a8e, "delete y_accum MEMSETs / hop waits")

## 1. What changed

Diff HEAD~1 -> HEAD (only `kernel.neff` changed; 4 members):

| member | slots | change |
|---|---|---|
| DVE0.bin | 14037 -> 14139 (+102) | -154 MEMSET (y_accum zero-init) offset by ~+256 inserted/moved EVENT_SEMAPHOREs (hopped S[3] waits, S[6] lag-0 waits on a fresh ES); 14087 slots differ because of the renumbering |
| Pool0.bin | 5283 -> 5283 | same count; 4135 slots differ = wait/inc value renumbering (the first y+=Cs ADD of each group becomes a 512-fp32 COPY) |
| Activation0.bin | 3040 -> 3040 | 2038 slots differ: only `events_semaphore_value` (S[6] wait counts, shifted by the deleted MEMSET increments) |
| PE0.bin | 10135 -> 10135 | 4958 slots differ: only S[6] wait values |

Against the unedited baseline the diff is cumulative: DVE 15953 -> 14163 instrs, Pool 3866 -> 5307.

This matches the commit message (MEMSET deletion, hopped waits, moved S[6] waits). I could not
independently attribute each of the 144/68/112 counts. `notes/` is empty (README only). The agent
did not record anything, so the next agent has no written history. Fix that.

## 2. Is the win real?

Yes. `profile_latest.json`: median 6603.96 us, min 6603.02, max 6604.95 (spread 0.029%, ~2 us).
Baseline: 7919.96 median, spread 0.024%. My own re-profile gave 6604.7. The step from the previous
commit is 6711.7 -> 6603.9 = 108 us, about 50x the spread. Not noise. Cumulative -16.6%.

## 3. Safety

- `semaphores.py` on the unpacked NEFF: for every semaphore max_wait <= total_inc (S3 3986/3987,
  S4 2555/2555, S5 5023/5023, S6 8022/8022; DGE and DMA queues balanced, same pattern as baseline).
  Nothing waits for a count that is never produced, and S[6] is still fully incremented.
  Note S[6] has 13462 waiters on 8022 incs and S[3] has 4483 on 3987. The renumbering
  scheme (newmap[k]) is the delicate part, since a wait value that is too low silently passes.
  Count balance does not prove ordering. The only evidence of ordering is that the output is
  bit-identical on one input set.
- Error: output_0 max_abs_err 0.12497 (83% of the 0.15 limit), output_1 1.81e-4. These are
  identical to `verify_baseline.json`, so the edit is bit-exact on this input. The 83% is
  pre-existing (bf16 ulp 0.125 at magnitudes 16-32), not caused by this edit. It does mean any
  change in fp32 accumulation order is dangerous: one flipped rounding at magnitude >= 32 gives
  a 0.25 error and fails.
- Hazard worth a stress test: 68 Cs buffers were moved into the SBUF hole at 62272 (buffer
  recycling edit). Fixed-input verification passes either way if a race is rare. Run verify and
  profile output comparison 5-10 times and with different random inputs (`tools/verify.py` seeds?).

## 4. Where the time is (measured, per-instruction table)

Work unit = (ctile 20) x (ftile 8) x (state 16) = 2560 units. Per unit:
ACT exp 0.72 us; PE 2 matmuls 0.59 us each; DVE: dBx = dtx*B_psum (TT 0.675 us), scan (1.31 us),
Cs = C_psum*state (TT 0.675 us); then y+=Cs (DVE 0.67 us or Pool 1.98 us).

Busy: DVE 6521 us (98.7%), Pool 5611 (85%), PE 2970 (45%), ACT 1903 (29%), SP 609, DMA 1443.
Timeline: DVE is 100% busy in every 0.5 ms bin from start to end.

Pool breakdown: 2260 TT = 4469 us; 1741 COPY = 732 us (837 are 1-element carry copies);
680 DMA_DIRECT2D (software DGE, dge:Pool queues) = 447 us.

DVE floor with all y-adds gone: 2560 x (2x0.675 + 1.31) = 6.8 us-nominal, ~6.3-6.5 ms measured with
instruction overlap. We are at 6.52 busy, so the remaining DVE-moveable work is only ~466 extra TT
(about 326 y-adds, 160 dtx=dt*x at line 182, 160 y+=D*x at line 297; ~300 us) plus ~60 us of 24 ns ESs.
The two PSUM-reading multiplies and the scan cannot leave DVE (Pool cannot read PSUM; only DVE
has TENSOR_TENSOR_SCAN). **The remaining headroom is small: realistic -1% to -3%, hard ceiling
about -4.8% from DVE/Pool balance (below).**

## 5. Ranked candidates

Balance model: moving one 512-fp32 TT from DVE to Pool saves 0.675 us DVE and costs 1.98 us Pool.
DVE 6521 - 0.675k = Pool 5611 + 1.98k at k ~ 343 -> ~6290 us (-4.8% at the very best). Freeing
Pool time raises k.

| # | Candidate | Detail | Ceiling / source | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | **Delete the 1-element carry COPYs on Pool** (Pool0, ~837 COPY at nki line 243, plus the ~104 per-line 168/286/305 ones are different - leave them) | Point the next-ftile scan's `initial` imm (`imm=[fp32@0x4110]` etc., TENSOR_TENSOR_SCAN operand) directly at column 511 of the previous scan's state tile for the same n, and drop the COPY (leave an ES to keep the Pool S[3] count; or renumber like iter-2). Final state still needs the last ftile copy to carry_all (keep those 20x16) | Frees up to ~0.37-0.5 ms Pool (COPY mean 420 ns incl. stalls; 837 x 438 ns = 367 us from profile). That frees ~190-250 Pool-slots of adds -> k+190 -> DVE about -130 us (-2%). Also removes S[3] hops that DVE ES waits on (2505 us of S[3] ES duration on DVE, though mostly overlapped). Bound: DVE/Pool balance above | The state tile buffer must still be live 16 units later (ftile k+1, same n). If the allocator recycles state_sb earlier it reads the wrong carry; fixed-input verify WOULD catch gross misreads (output wrong), a rare race wouldn't | Read the allocation: `dst` addresses of the 16 consecutive scans (0x6940, 0x9940, ...) - if any of them is reused within 16 units the idea dies. 5 min with the Instruction table |
| 2 | **Move remaining DVE-side TTs to Pool as capacity allows**: ~326 y-adds still on DVE, 160 `dtx=dt*x` (line 182), 160 `y+=D*x` (line 297) | Pool TT at 1.98 us. Pick ones where the deadline slack (DVE slots until consumer) is >= 3, since iter-3 found 151 adds stalling DVE when slack was only 3 slots | Up to ~-300 us DVE theoretically; bounded by Pool: needs ~1290 us Pool, only ~1000 us idle and that is fragmented (Pool gaps 1056 us in 1465 gaps) -> alone maybe -50 to -100 us; combined with #1 -150 to -250 | Bit-exactness: the add order must be preserved (y accumulation chain); the D*x add and dtx are single ops - bit safe. Race risk as in #1 | Move only the 160 D*x adds first (no chain order issue), profile: if Pool busy > 90% and total not down, stop |
| 3 | **Move Pool software-DGE DMA descriptor generation to SP** (680 DMA_DIRECT2D = 447 us of Pool) | Re-queue the B/C row loads (the dge:Pool S[8..23] queues) onto SP-issued hardware DMA (SP is 9% busy) via the SP0.json / DMA descriptor path | Frees up to 447 us Pool -> k+225 -> maybe -150 us. This is an upper bound only; mechanism not shown | Medium-high: new DMA queue/semaphore plumbing; docs ch. 09/13 only verify re-pointing an existing queued DMA, not changing the issuing queue | **Unknown mechanism**: first try to find in `neff_doc/docs/09-dma.md` whether a queue's engine can be changed; if not, kill it |
| 4 | **Fold the 24 ns ESs into the adjacent instruction's own wait field** (DVE ES count 5954; ~2380 of them are <=100 ns) | Where the next instruction has a free wait slot, move the ES's wait there and delete the ES, keeping S[6] increments | ~2380 x 24 ns = 57 us at most (<= 0.9%), probably 20-30 us. Bound: DVE ES <=100 ns total 62.9 us from profile | Low-medium: fewer waits = loosened sync if mis-moved; keep each wait on the same or a later instruction that is the consumer | Count how many ES have a following instruction with an unused wait; if < 500, kill |
| 5 | **PE fp32 accumulation of y** (PSUM-accumulating matmul with identity, ACT drains) | Would take the 2260 Pool adds (4.47 ms) away, letting Pool absorb DVE ops | Unknown; bound: PE has 3.6 ms idle but an fp32 matmul is 4-pass (~1.5-2.3 us each), so it can host at most ~1000-1500 adds. Gain on paper -5..-10% | **High**: changes fp32 summation order -> not bit-exact. Error is at 83% of the limit; a single flipped bf16 rounding at |y| >= 32 fails (0.25 > 0.15). PSUM bank pressure (one 2 KB bank per live accumulator). New matmul/ACT instructions in a fixed-slot NEFF = large structural edit | Run a numpy model of the golden in fp32 with permuted/pairwise 16-term sums and see how often the bf16 output flips and where max_abs_err lands. If > 0.15, dead |

Ranking logic: #1 is a deletion with a checkable lifetime precondition and the largest realistic
gain per risk; #2 is a continuation of the proven mechanism, gated by #1; #3 is bigger but
unproven; #4 is safe but tiny; #5 is the only one that could break the bottleneck but threatens exactness.

## 6. Closed (do not redo)

- Moving *all* y+=Cs ADDs to Pool: over-subscribes Pool; iter-3 had to pull 151 back (183 us of
  DVE stalls in 151 gaps with only 3 DVE slots of slack). Pool TT is 1.98 us vs 0.67 us on DVE (2.9x).
- Zero-init MEMSET for y_accum: deleted this iteration (154 MEMSETs; 6711.7 -> 6603.9).
- Merging the 1-element carry COPYs pairwise into dim-1 strided copies: done in iter-2
  (-300 us of Pool time). Do not do further merging; #1 deletes them instead.
- Removing waits on an engine's own semaphore: verified negative (docs ch. 10.5).
- Moving the PSUM-reading multiplies, or the scan, off DVE: Pool cannot read PSUM, and only DVE
  has TENSOR_TENSOR_SCAN (no Pool/ACT scan in the opcode table). 2x/4x DVE modes need bf16 operands, which would change
  the numbers (state and dtx must stay fp32).
- Noise: spread 0.03%; the loop is not chasing noise.

## 7. Unknown

- Whether state_sb tiles are live 16 units later (decides #1). Measure: addresses of dst of each
  scan across 17 consecutive units.
- Real fp32 matmul cost on PE and whether fp32 PE accumulation is exact (decides #5). One
  micro-NEFF or the `neff_doc` experiments.
- Whether Pool's DGE queues can be re-issued from SP (decides #3): docs/09-dma.md.
- Whether the 83% error is a single bf16 ulp at magnitude 16-32. Check where the max error lands
  in `verify` output; it sizes the risk of any non-bit-exact edit.
- Whether a ~5% remaining gain is worth it: the kernel is DVE-bound by ~6.3-6.5 ms of
  unavoidable PSUM-fed multiplies and scan. After #1 + #2 (-3% at best), I would call the rest
  irreducible at the NEFF level.
