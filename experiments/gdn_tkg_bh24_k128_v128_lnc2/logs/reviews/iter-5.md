# Review of iter-5 (`kernel.neff`, gdn_tkg bh24 LNC=2)

Note: `baseline/` is the original 76.05 us NEFF, so `neff_diff baseline kernel.neff` shows the cumulative work of
iters 1-4 (12 members). This iteration's own change is `git show HEAD:kernel.neff` vs the working `kernel.neff`.
`notes/README.md` is still the empty template: the agent wrote no notes, so "what the notes claim" is only the
commit-log text for earlier iterations. Nothing was claimed for iter-5.

## 1. What changed (HEAD -> working tree)

6 members changed: `sgNN/{Activation0,DVE0,PE0}.bin`, same in both subgraphs. Pool/SP/json untouched. No instruction
was moved, added or deleted. Only sync fields differ.

* ACT slots 22, 25, 37, 54 (the four CASTs `S->S_bf`, opcode 0x47) went from `events_update_mode NONE` to
  `SEM_INC_COMPLETE`, `update_idx 4`. ACT now increments sem 4 41 -> 45 times.
* Every wait on sem 4 that sits after one of those slots had its `events_semaphore_value` renumbered upward:
  ACT 7, DVE 12, PE 10 slots. The two subgraphs are identical in every slot and value.
* Four PE LDWEIGHTS waits (PE slots 20, 22, 36, 60) were retargeted to an earlier producer, not just renumbered. I
  decoded which ACT instruction each wait value points at:
  * PE20: ACTIVATION_READ_ACCUMULATOR(24) -> CAST(22)
  * PE22: ACTIVATE(26) -> CAST(25)
  * PE36: CAST(38) -> CAST(37)
  * PE60: CAST(56) -> CAST(54)

  The PE waits now sit on the CAST that writes the bf16 tile the next MATMUL reads (PE21 reads 19824 = CAST22's dst;
  PE23 reads 16752 = CAST25's dst; PE37 reads 16752 = CAST37's dst; PE61 reads 17264 = CAST54's dst). Same pattern in
  sg00 and sg01. So the change is "give the S_bf CASTs their own completion count so the PE can start as soon as its
  own tile is cast, instead of waiting for the next ACT instruction". This is a pure sync tightening to the true
  producer. The arithmetic is untouched and error is identical to baseline.

## 2. Is the gain real?

* Agent's own run: 38.396 us median, spread 0.68%, samples 38.21-38.47.
* I re-measured both on the same device in the same session, 5 repeats each: previous HEAD NEFF 38.720 us, working
  NEFF 38.185 us.
* Delta is -0.5 us (-1.4%), about 2x the run-to-run spread (~0.25-0.3 us). It is small but real, because both
  measurements agree in sign and size. It is not a large win: any further sync-only trimming of this kind is now at
  the noise floor. Accept it, but do not keep scoring sub-0.3 us moves.
* Cumulative: 76.05 -> 38.2 (-49.8%).

## 3. Safety

* `semaphores.py` for sg00 and sg01 gives identical tables. Sem 4: max_wait 45 = total_inc 45. Sem 5: 38/38. Sem 6:
  79/79. DMA sems 9-16 and 24-27 are all balanced. No wait exceeds its increments, and no wait value was lowered below
  the true producer count except the four deliberate retargets above.
* Residual race risk, low. The retargeted PE waits no longer order the PE after ACT 24/26/38/56. I checked that the
  MATMUL source buffers match the CAST dsts. I did not trace the LDWEIGHTS sources (219646, 219902, ...). They are
  SBUF 0x359xx-0x35Bxx, outside the S_bf tile addresses, so the old ACT waits looked like conservative
  over-waiting. The WAR on 16752 (CAST25 then CAST37 write the same tile) is still protected by the existing
  PE->ACT sem 5 waits, which are unchanged.
* The two cores cannot disagree: edits are byte-identical in the two subgraphs and each subgraph has its own
  semaphores.
* Error: output_0 max_abs_err 2.322e-2 is 86% of the 2.7e-2 limit, but it is exactly the baseline value (2.3223e-2).
  This is bit-identical arithmetic, and there is no new error to worry about. Any candidate that changes dtype or
  function tables would eat into the remaining 14% and needs checking.
* Verifier limits: it runs one fixed input and one run. It cannot catch a wait that is too loose but happens to be
  satisfied on that run. Only the semaphore accounting plus the producer/consumer address check can, so repeat that
  check for every sync edit.

## 4. Candidates for the next iteration (ranked)

Where the time goes (profile pcore 0; sg01 is the same within 0.5 us; kernel body is ~12.0 -> ~49.1 us on the
profile clock, then ~3 us fixed postamble):

* ACT head, 12.2 -> 23.9 us, is a serial chain. It runs 7 ACT_TABLE_LOADs back to back at 1.283 us each (8.98 us
  total) with only 0.2-1.5 us of real activation between them, and ACT wait times on the loads are ~2 ns.
  PE and DVE are idle behind it (DVE gaps 1.7+1.8+1.9 us, PE gaps 5.3+5.0 us).
* The 7 loads are, in order, tables 1, 0, 2, 0, 4, 3, 2. Mapping from the ACTIVATE func ids: T1 = softplus (func 9),
  T0 = exp (func 7), T2 = reciprocal_sqrt (func 29; func 30 = square, which exists in every set so it never needs
  its own load), T4 = silu (func 36), T3 = sigmoid (func 5). Exp (T0) and rsqrt (T2) are each loaded twice.
* Body, ~27 -> 49 us: 12 repeats of PE matmul -> ACT `square+accumulate` (0.26 us) -> READ_ACCUMULATOR (0.29 us) ->
  ACT rsqrt (0.2 us) -> DVE. ACT idles 0.9-1.9 us in 5 gaps waiting on PE/DVE, so the body is a latency chain, not
  ACT-throughput bound.
* `engine_busy` of ACT 89% and DVE 92% is inflated: it includes the EVENT_SEMAPHORE wait spans (ACT 8.6 us, DVE 6.7 us
  per core), so true compute is ~28-30 us each. Pool has ~24 us idle.
* Tail: last DVE op ends ~49.1 us, SP triggers the output DMA (`WRITE S6>=79`), and it completes ~50.4 us. That is
  ~1.3 us of DMA latency, then the fixed postamble.

| # | Candidate | Where | What changes | Mechanism / evidence | Gain, measured ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Drop the redundant T0 (exp) table reload** (ACT slot 8). Optionally also the 2nd T2 (slot 19). | `sg00/Activation0.bin` and `sg01/Activation0.bin`, slots 3/4, 8/9 (and 19/20) | Move ACTIVATE slot 9 (exp, 12 elems, src 0x35700) to sit directly after slot 4, then delete slot 8's `ACTIVATION_TABLE_LOAD`. Replace the removed slot with an `EVENT_SEMAPHORE` that keeps ACT's sem-4 count (docs 14.3 #3 / ch.10 R4) and renumber the sem-4 waits (DVE, PE) as this iteration did. The DVE `STREAM_SHUFFLE` of 0x35700 (profile 18.16 us) must be hoisted ahead of the one for rsqrt output 0x4128. | The table-load cost is measured (1.283 us each). ACT is waiting on nothing during the head, so the saving lands straight on the chain. Load order/func ids come from decoding the ACT stream. | **<= 1.28 us** from the exp reload (profile: ACT_TABLE_LOAD 8.98 us/core, 7 x 1.283). A second dedupe of T2 could add up to another 1.28 us but needs rsqrt(slot 7) to move after silu/sigmoid, and slot 7 feeds the exp input, so treat it as a stretch. Realistic: 0.8-1.3 us (2-3%). | Medium. The wrong data flow shows up as a wrong `g`, which the verifier would catch. A race on the moved shuffle might not be caught, so check DVE writes of 0x35700 in the window (the `SCALAR_TENSOR_TENSOR` at 15.99 us may write it). | Read the DVE stream: is the data at 0x35700 (and its shuffled copy 0x835700) final before ACT slot 7 completes? If something between writes it, the candidate is dead in 2 minutes with no device run. |
| 2 | **Shrink the rmsnorm chain on ACT by batching rsqrt** | `Activation0.bin` both sgs, the 12 repeats of (FUNC_30 accum, READ_ACCUMULATOR, FUNC_29) | Point the READ_ACCUMULATOR `dst` of 2-4 consecutive heads at adjacent fp32 slots and merge their FUNC_29 ACTIVATEs into one with `num_elem` 2-4 (ACT 0.2 us each). | Fewer, larger instructions (docs 14.3 #4). | Ceiling **<= ~1.2 us of ACT time** (6 merged x 0.2 us), but the ACT gaps show the body is chain-latency bound, so the realistic wall gain is perhaps 0.3-0.6 us and may sit inside noise. Also delays the first consumer of each merged group. | Medium. Changes pairing of waits on DVE sem 6 consumers. The verifier would catch a wrong address, not a missed ordering. | Check that the 12 READ_ACC dsts (0x4000, 0x4008, 0x5070, 0x35c00, ...) can be made contiguous without clobbering their consumers' sources, and that each FUNC_29's consumer is on one engine. If not, drop it. |
| 3 | **Offload CAST / TENSOR_TENSOR work to Pool** | Move some of the 14 ACT CASTs (5.1 us/core, ~0.28-0.3 us each) or DVE ops into `Pool0.bin` | Rewrite the slot's engine stream per docs 14.3 #3: leave an `EVENT_SEMAPHORE` on the old engine, insert the op on Pool with the same waits, point consumers at the Pool sem count. CAST / COPY / TENSOR_TENSOR_ARITH are documented as running on Pool. | Precedent: commits 400b689 and 119bfcf gained 19% and 9.5% by moving ACT work to DVE. Pool is ~24 us idle in the body (profile: Pool issues only 4 state-out DGE DMAs from 31 us on). | **Unknown.** Pool per-element speed for fp32->bf16 is not measured; an optimistic bound is the ACT gaps (0.9-1.9 us x 5) if the offloaded op is on the critical path. | High effort: new Pool sem counts, Pool also issues the DGE DMAs (so a long Pool CAST delays the state-out DMA triggers). Verifier catches wrong values, not a lost wait. | Retarget ONE ACT CAST (e.g. slot 22) to Pool and read its duration in the profile. If it is > ~0.6 us or it stalls the DGE, stop. |
| 4 | **Split the final output DMA tail** | `SP0.bin` (the `WRITE`s at 49.1 us gated on DVE sem 6 >= 79) and the DVE op that gates it | Split the last tile's output descriptor into two and trigger the first half after its own DVE op instead of after the final one. | Tail from last DVE op (~49.1) to DMA complete (~50.4) is ~1.3 us (profile). | <= ~0.5 us. DMA fixed latency is part of that 1.3 us and cannot be removed. | Medium; needs new descriptors and a DMA-queue count change (+ matching semaphore waits). | Check in `DmaPacket.parquet` how long the final `o` transfer is in flight beyond fixed latency. If the transfer itself is < ~0.3 us, close it. |

Rank reasoning: #1 has a measured per-unit cost, a static pre-check that needs no device, and the highest ceiling
per effort. #2 and #4 have small, uncertain ceilings. #3 has the largest unknown upside but the highest effort, so
only go to it after the one-instruction probe.

### Closed (do not redo)

* ACT->DVE move of the 36 scale-pointer ACTIVATEs, the wide S->S_bf cast, hoisting the 12 silu ACTIVATEs into one
  wide op, deleting the 12 `o_sb` COPYs, the 4-way state-out DMA split, and the S*=exp_g/cast pipelining: all already
  done (iters 1-4, -49.8% total, 76.05 -> 38.2).
* ACT_TABLE_LOAD count 22 -> 7 per core: already cut. The remaining headroom is the duplicate exp (T0) and rsqrt
  (T2) loads only (candidate 1). Do not hunt for more than that in the table plan.
* Sync-only retargeting of PE waits to the true CAST producer (this iteration): worth -0.5 us (38.72 -> 38.19).
  This is the noise floor; further sync tightening of this type will not be distinguishable from noise.
* Changing dtype / function tables to be cheaper: not attempted; output_0 error is at 86% of the limit
  (2.322e-2 / 2.7e-2), so there is little room.

### Unknown (measure before ranking changes)

* Pool engine speed for CAST / TENSOR_TENSOR on these shapes. Settle it by moving one CAST to Pool (candidate 3
  probe).
* Whether exp#2's input (0x35700) is final early enough to hoist (candidate 1 pre-check, static).
* How much of the 1.3 us output-DMA tail is fixed latency vs transfer time. Settle it with `DmaPacket.parquet` for
  the final `o` transfer.
* Whether the first input-DMA arrival (~14.0 us, 1.8 us after kernel start) can be earlier by reordering the
  descriptors of `a`/`A_log`/`dt_bias`. The DVE and ACT head chain waits on it; `z` was already moved first in
  119bfcf. Settle it with `DmaPacket.parquet`: per-descriptor completion times in the first 4 us.
* Whether there is any further hidden reorder gain in the PE idle gaps (5.3 + 5.0 us in the head): PE is idle only
  because ACT's head chain feeds it; fixing the head (candidate 1) is the lever, not the PE stream.
