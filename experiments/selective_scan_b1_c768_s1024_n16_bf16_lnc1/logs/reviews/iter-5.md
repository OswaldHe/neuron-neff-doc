# Review of iter-5 (working-tree `kernel.neff` vs HEAD af2c8b3 = 508.8 us; baseline 602.64 us)

Sources: `neff_diff.py HEAD kernel.neff`, `profile_latest.json`, `verify_latest.json`, `semaphores.py` on unpacked HEAD and working copies, and my own `profile.py --repeats 1 --parquet` run (501.338 us). `notes/` contains only the README, so there is no written claim to check against. There is also no commit message yet. I compare the diff against the HEAD message and against what the diff itself shows.

## 1. What changed

3 members changed (Activation0, DVE0, Pool0). Everything else is identical.

| member | slots | change |
|---|---|---|
| Pool0.bin | 389 -> 389, 21 differ | The ct0 carry COPYs (16 slots, dst `0x4100+4n`) now write `0x16000+4n`. The ct0 final-state store, `PSEUDO_DMA_DIRECT2D` slot 72 (64 B), now reads src `0x16000` instead of `0x4100`. One carry COPY (slot 48) has its src moved from `0x513c` to `0x168fc` (see below). |
| DVE0.bin | 1057 -> 1055 slots, 483 differ | Every scan `imm0` (the initial-carry pointer) that read `0x4100+4n` now reads `0x16000+4n`. One MEMSET initialises `0x16000` (16 elements). The group-0 scan at slot 97 has dst `0x4940` -> `0x16100`, and the TT that reads it has src1 `0x16100`. Two `EVENT_SEMAPHORE` slots were dropped: old slot 96 (wait S9>=32) and old slot 177 (wait S11>=32). Slot 96 survives with its S6 extended wait only. That accounts for the 2-slot shrink; all other differences are index shifts. |
| Activation0.bin | 234 -> 234, 24 differ | An ACT_TABLE_LOAD and an ES swapped places, and some S6/S4 wait values were renumbered. |

What it is: a ping-pong of the SBUF homes of two ct0 buffers, the 16-float carry and the group-0 y scratch. Both were read by an in-flight store DMA (Pool DMA at slot 45 stores 2048 B from `0x4940`; Pool DMA at slot 72 stores 64 B from `0x4100`) and overwritten by DVE. DVE therefore had to wait for those stores to finish (S9>=32 and S11>=32). With the writers on fresh addresses the two WAR waits are dead and were deleted. I checked every instruction operand in the profile: `0x16000..0x1603f` and `0x16100..0x168ff` are touched only by the edited slots. The Pool COPY src `0x168fc` is the last element of the new y scratch. The carry source for the next chunk follows the moved scan dst, so that is consistent.

Caveat. I verified the mechanism for the slot-96 wait: the scan at slot 97 had dst `0x4940`, which DMA 45 reads. I verified the slot-177 wait by the carry/DMA-72 retarget. I did not grep `def.json` / DMA descriptors for the `0x16000..0x16900` hole. Only instruction operands were checked.

## 2. Is the latency change real?

Yes. Medians:

| run | median | spread |
|---|---|---|
| `profile_latest.json` | 501.414 us | 0.025% (501.345 .. 501.468) |
| my re-profile | 501.338 us | single repeat |
| HEAD | 508.8 us | |
| baseline | 602.637 us | 0.052% |

The step is -7.4 us, about 60x the 0.13 us sample range. Cumulative is 1.2021x. The mechanism accounts for it: two DVE stalls of about 3.5 us each on store-DMA completion, now gone. The loop is not chasing noise at this size. It is, however, running out of stalls to remove (see section 4).

## 3. Safety beyond the verifier

- Output error is unchanged: `output_0` max_abs_err 6.245e-2 (83% of the 0.075 limit), `output_1` 4.27e-5. Both are identical to `baseline/verify_baseline.json`, so the numbers are bit-identical. The 83% is pre-existing bf16 ulp error; any change in fp32 accumulation order would be dangerous.
- `semaphores.py` on HEAD vs now: only the waiter counts on S9 and S11 drop (8 -> 7 each). Every `total_inc >= max_wait` still holds (S3 282/282, S4 194/194, S5 374/374, S6 640/640). S15 and S18 have max_wait 48 against 64 increments, as before. The tool does not read the extended wait field, so it would not catch a deadlock hidden there. The NEFF ran to completion, so no deadlock.
- Race analysis. The removed waits protected WARs on store sources (`0x4940`, `0x4100`). Both sources are now written only by instructions that no longer touch them, and the store DMAs read the old buffers or, for DMA 72, the retargeted `0x16000`. DMA 72 was updated, so the output_1 path is consistent. Residual risk: the fresh hole has other readers or writers I did not check in `def.json`. A fixed-input verify cannot prove WAR/RAW correctness, so run `verify.py --runs 3` and one different-seed input before building on this.

## 4. Where the time is now (profile of the current NEFF)

| engine | busy | util | what |
|---|---|---|---|
| DVE | 481.8 us (union 494.7 us of 509) | 96% | 192 scans x 1.28 = 245.7; 384 PSUM-source TT x 0.67 = 257.5; 36 SBUF-only TT = 35.0 (24 fp32 `y+=` ADDs avg 0.84, 12 bf16 MULT avg 1.24); 32 MEMSET 2.6; ES overlap about -77 ns each |
| Pool | 417 us | 83% | 168 TT ADD (median 2.16 us, 338 us); 115 COPY (38 single-element + 77 two-element carries, 35 us); 60 DMA_DIRECT2D x 657 ns (39 us); gap 91 us in 89 gaps averaging about 0.95 us |
| PE | 218 us | 44% | 384 matmuls x 0.59 |
| ACT | 146 us | 29% | 204 SOFTPLUS/EXP x 0.72 |

- DVE runs gap-free from its first compute (14.3 us) to its last (507.3 us). The only DVE gaps above 150 ns are the head (a 4.2 us wait for the first input DMA, S8>=16) and a handful of fixed runtime items (TENSOR_LOAD 2.0 us, tail RANGE_CLEAR 3.4 us). Sync-wait stalls are no longer a source of gain. The kernel is DVE-throughput-bound: wall about 501 us against DVE continuous 493 us.
- Pool TT ADD duration is bimodal: 134 of 168 run at 2.16 us and 19 at 1.27 us. The fast ones cluster at group boundaries where DVE is running boundary work rather than back-to-back scan+TT. Pool pays about 0.9 us per ADD when DVE runs scan+TT. That is consistent with a shared-SBUF-port penalty, but I did not isolate it.
- Reducible DVE content is small. The 384 PSUM-reading TTs and 192 scans (503 us raw) cannot leave DVE: Pool cannot read PSUM, and only DVE has `TENSOR_TENSOR_SCAN`. Only the 36 SBUF-only TTs (35 us raw, about 22 us after the ES overlap) and the 12 first-writer COPYs (5 us) are movable. The realistic remaining gain is 2-3%; the hard ceiling is about 5%.

## 5. Ranked candidates for iter-6

Balance model from this profile: DVE 482 - 0.68k = Pool 417 + 2.16k gives k about 23 ops (about -15 us). Pool capacity is the cap, not DVE.

| # | Edit | What changes and where | Mechanism / source | Gain and ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|
| 1 | **Move DVE-side fp32 `y+=` ADDs to Pool, starting with the middle groups.** | DVE0: the 24 SBUF-only `TENSOR_TENSOR ADD` slots with src0==dst (12 group-boundary `y+=Cs`, 12 more `y+=` ADDs). Pool0: same ADD template as the 168 already there (Pool0 slot 235 style). Hop the consumer's wait to a later Pool-S3 count and re-point any recycled buffer to the free SBUF hole (`0x16900..0x1c000`, about 22 KB, 11 x 2 KB tiles). | Iter-1 recipe of this kernel (170 ADDs, -86 us) and this iteration's retarget trick. Established on the device. | Gross 24 x 0.68 = 16 us DVE. Pool +52 us -> 94% busy. Realistic **5-10 us**. Cap: Pool 83% now. Skip the last group's final ADD (it lengthens the tail). | Medium. WAR/RAW at the group boundary (y store DMA reads the accumulator). The verifier will not catch a hazard that never fires on this input. | Move 4 ADDs from groups 2-5. Pass if DVE busy drops >= 2.4 us and the median drops >= 2 us. Fail if Pool grows a new gap before a DVE consumer. |
| 2 | **Fuse the two 512-element scans of each (channel-tile, state) into one 1024-element scan.** | DVE0: 96 pairs of `TENSOR_TENSOR_SCAN` slots -> 1 slot each with num_elem 1024. Delete the 115 Pool carry COPYs and the S3 waits on them. dA (ACT dst, adjacent already: `0x5140`, `0x5940`), dBu and h need adjacent placement. | The carry does not reset across AP dims (c2560 experiment iter-1: `[256,2]` bit-exact). | Unknown per-scan fixed cost. If about 0.22 us (c2560 probe): 96 x 0.22 = **21 us DVE**, plus **35 us Pool freed**, which unlocks #1 and #3. If the cost is about 0.05 us: about 5 us. | High. Layout change on every operand, and the C-multiply of chunk 1 is deferred, which can stall the PE PSUM bank. | Edit one scan slot to num_elem 1024 (garbage output, timing only; no verify). If duration < 2 x 1272 - 150 ns, continue; otherwise drop. 10 minutes. |
| 3 | **Re-issue the Pool software-DGE loads from SP.** | Pool0: 60 `PSEUDO_DMA_DIRECT2D` x 0.657 us = 39 us. Re-queue the B/C row loads (the `dge:Pool` queues S8..S23) onto SP (9.6% busy). | Unknown. Docs ch. 09 / 13 show re-pointing an existing DMA, not changing the issuing engine. | Up to 39 us of Pool -> about 18 more ADDs -> about 12 us DVE. Realistic about 0 until the mechanism is proven. | Medium-high (new queue/semaphore plumbing). | Read `neff_doc/docs/09-dma.md` for whether a DMA instruction can live on SP with `PSEUDO_DMA_DIRECT2D`. If not, kill. 15 minutes of reading. |
| 4 | **Shorten the head.** DVE first compute at 14.3 us, after a 4.2 us wait on the first input DMA (S8>=16). | Pool0 slot 2 (first DGE load) and DVE slot 5 wait. Issue the first load earlier, or split it so DVE's first TT waits only on a half-size tile. | Unknown. | Ceiling **4.2 us** (measured wait). Realistic 1-2 us. | Low-medium. | Check whether the DMA issue is itself delayed by an earlier Pool wait: profile `start_ts` of slot 2 against the 5.7 us Pool start. If Pool slot 2 starts at its first possible moment, kill. |

Rank reasoning: #1 has a proven recipe and a modest but certain cap. #2 has the largest ceiling and unlocks Pool capacity, but the structure is invasive; its probe costs minutes, so run the probe first. #3 and #4 are small and uncertain.

If #2's probe says the fixed cost is small, the remaining realistic total is 2-3% (10-15 us) and the kernel is effectively irreducible at the NEFF level.

## 6. Closed (do not redo)

- **Sync-stall removal on DVE.** DVE idle is 14.4 us in total, of which about 12 us is fixed (head wait, runtime TENSOR_LOAD, tail RANGE_CLEAR). Nothing left to gain from loosening waits.
- **DVE EVENT_SEMAPHORE slots are free**: -77 ns overlap with or without an ES between compute instructions (c2560 experiment iter-4 review, same pipeline). Removing satisfied ESs gains nothing.
- **PSUM source penalty / bf16 speedup**: PSUM-source TT 0.653-0.68 us, SBUF-only fp32 ADD 0.68 us (HEAD message: 677 ns). No penalty, no bf16 fast path on DVE. An ACT PSUM->SBUF evacuation gains nothing.
- **Moving PSUM-reading multiplies or the scan off DVE**: Pool cannot read PSUM; only DVE has the scan opcode.
- **Pointing the second chunk's scan imm directly at the previous scan's h tile** (deleting the carry COPY): the h tiles recycle in a ring of about 3 (scan dst `0x6940` repeats at unit 1 and unit 4), not 16. The lifetime precondition fails. The hole holds about 11 tiles, not 96.
- **Zero-init MEMSET removal / ADD->COPY first writer / Pool carry pairwise merge**: already done (commits 5e1f980, af2c8b3, 6e7af17).
- **PE fp32 y accumulation**: bf16-only matmul microcode; also changes summation order against an 83% error budget.

## 7. Unknown (would change the ranking)

- Cost per scan beyond element count (`num_elem` 1024 probe, see #2). It decides whether #2 is worth its layout work.
- Why 19 of 168 Pool ADDs run at 1.27 us versus 2.16 us. If it is SBUF-port contention with specific DVE ops, steering Pool ADDs onto DVE-idle windows would raise Pool capacity by up to 1.7x (k about 23 -> about 40 ops). Measure: put one Pool ADD in a window with DVE stalled and compare. Not tried.
- Whether `def.json` / DMA descriptors reference `0x16000..0x1c000`. `grep` the unpacked `def.json` variable table before allocating more tiles there.
- Whether the Pool DGE queue can be re-issued from SP (#3).
- Whether the 12 bf16 MULT (avg 1.24 us, 693 ns when unstalled) can be hidden behind the head-of-kernel DMA wait.
