# Review of iter-0 (gdn_cte_b1_s128_d128)

## 1. What changed
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports an identical header, 34/34 identical members and "payloads are identical". `cmp` confirms the files are byte-equal, and `git status` is clean. `notes/` holds only the README template, so there is no claim to compare against. There is no `profile_latest.json` or `verify_latest.json` in the worktree.

## 2. Is the latency change real?
There is no change to measure. I re-profiled the unedited NEFF as a control (3 repeats): 48.410 / 48.802 / 48.769 µs, median 48.769, against a baseline median of 48.519. That is +0.5% from the same bytes. The baseline's own 7-capture spread is 48.213–49.157 µs (1.9%).

- Anything under about ±0.5 µs on this kernel is noise.
- A win has to be re-measured at least twice, with the baseline re-measured in the same session.
- A 1.3% spread on a 49 µs kernel is ±0.3–0.6 µs, so a "-0.3 µs" result needs two or more re-measurements before it is believed.

## 3. Safety
Nothing was edited, so no synchronisation was loosened. `semaphores.py` on the unpacked baseline reports no over-wait. The verifier error is the baseline's. The error margin for this case has not been read; check `baseline/verify_baseline.json`. In the sibling s256 kernel, output_1 sat at 83% of the limit.

## 4. Candidates for next iteration

Evidence base. I did not run a critical-path walk on this NEFF. The structure is the same kernel as `neff_doc/experiments/gdn_cte_b1_s256_d128_lnc1`, which went 67.5 → 53.2 µs (-21%) over 7 iterations, with fusion of the staging round-trips worth -8.3% in its first iteration. `def.json` here has the same five `tmp-buf` vars (var8 16 KB; var9, var10, var11, var12 64 KB each). They are smaller than at s256, where they were 32 KB and 128 KB.

Profile facts for this NEFF, in µs from Pool[0] start:
- No engine is above 48% busy (ACT 23.4, DVE 19.3, DMA 17.9, PE 12.1, Pool 11.0, SP 6.5 of 48.5). The kernel is bound by a DMA dependency chain, not by an engine.
- ACT hosts DGE saves at 20.5, 22.6 and 23.2 (slots 14, 20, 21). SP hosts the matching reload DGEs at 22.6–25.5 (slots 4, 5, 6, 8).
- ACT hosts the var11/var12 saves at 43.1–45.8 (slots 56, 57, 59, 61).
- Pool slot 16 (reload) issues at 46.5 and SP slot 24 (reload) at 47.3.
- Tail: Pool store DGEs at 53.0 (slot 17), 58.1 (slot 19) and 59.2 (slot 20), then the final fence runs to 62.9.
- Each DGE costs 0.55–0.77 µs of its host engine.
- The head, 12.6 → first compute, is gate-load latency.

Treat the ceilings below as hypotheses taken from s256, scaled down. They are not measured on this NEFF.

| # | Candidate | Where | Change | Mechanism (source) | Expected gain / ceiling | Risk, and would the verifier catch it? | Cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | Fuse the reload-side staging round-trips for var8, var9, var10 (SBUF→DRAM→SBUF becomes SBUF→SBUF) | ACT0 save slots 14, 20, 21; SP0 reload DGEs 4, 5, 6, 8 (SP0.json also changes) | Rewrite the reload DGE source to `SB` (var_id 0) with step `(s//E)*S` and offset `So + …` (recipe in s256 iter-1 notes §2a). Retarget the reload's wait from the save's completion semaphore to the save's own wait. Replace the dead save with an `EVENT_SEMAPHORE` that keeps its wait. | s256 iter-1 §2a and §3.1: an SB→SB DGE works, bit-identical. Fusing var8 alone gave +0.2%; the gain only appeared once var9 and var10 were done too, so do all three together. | s256 saw about -3% from var9+var10. Here at most about 1.5 µs; the ceiling is the save-to-reload latency (ACT save 20.5 → SP reload 22.6–25.5 → consumer), unmeasured. | Wrong wait retarget gives a race, but the verifier would likely catch it as wrong numbers or a hang. A race that passes the fixed inputs is possible; run `verify.py --runs 3`. | Fuse only var9 and var10 and measure against 3 baseline captures. If the change is under 0.3 µs, drop it. |
| 2 | Fuse the save-side staging for var11 and var12 | ACT0 saves 56, 57, 59, 61 (43–46 µs); reload Pool0[16] and SP0[24] | Rewrite the 2-D saves to write SBUF directly (dst step `(s//M)*J`, offset `D+F/ratio`, `dst_elem_size /= ratio`, `out_dtype` taken from the reload; ratio 2 for the fp32→bf16 case). Turn the reloads into `EVENT_SEMAPHORE`s that keep their wait and `SEM_INC_COMPLETE` the old semaphore. Lower the consumer's wait from 16 to 1. | s256 iter-1 §2b: bit-identical, 9 DGEs rewritten. Worth -4.7% and -7.1% in steps 3–4 of s256 iter-1. | This sits on the 43–55 µs path into the tail. Plausibly 1–2 µs. Ceiling: reload issue (46.5/47.3) → consumer start. | Keep the compiler's fence slots untouched. SP's fence carries a second wait in the extended events block, and `semaphores.py` ignores that block, so write your own scan. | Apply to var12 only; if the end time does not move after 3 captures, stop. |
| 3 | Rebalance DGE host engines on the head and tail chain | ACT0 DGEs 56–61 and Pool0 DGEs 17, 19, 20 → move to SP (6.5 µs busy, 13%) | Move DGE slots to SP; the DGE keeps its own `semaphore` field. Prefer overwriting existing no-op `EVENT_SEMAPHORE`s over inserting, so the slot indices stay fixed. | s256 iter-1 §3.2: a DGE keeps its semaphore on another host engine, verified on the device. Cost is 0.55–0.77 µs per DGE on its host. | ACT has 4 DGEs on the 43–46 µs path, about 2.4 µs of issue. The move frees ACT, but only helps if ACT is binding. Likely ≤0.5–1 µs. | Low. The semaphore ids stay the same. | Move one ACT DGE to SP. If the end time is unchanged, ACT is not binding; stop. |
| 4 | Move the three Pool tail store DGEs (slots 17, 19, 20) to SP | Pool0 17/19/20 → SP0, using no-op `EVENT_SEMAPHORE`s | Same mechanism as #3. This tests whether the "Pool DGE costs about 0.5 µs of start-up" effect (s256 §8) is real. | s256 §8 hypothesis. Hoisting the wait there (edit7) was refuted at +0.4–0.7 µs, but the SP-move test was never run. | Ceiling about 0.5 µs per store; the tail is the 53–62.9 µs chain. | The verifier would not catch a store that moved past a fence it relied on. Keep the same waits. | Move slot 17 only; it is a 15-minute test. |
| 5 | Run a critical-path walk on the current NEFF before committing to #1–#4 | n/a | Build the backward walk described in s256 iter-1 §4: candidates are the previous slot on the same engine, plus the first time each wait was satisfied, with the semaphore reach times from `SemaphoreUpdate.parquet`. Join on `compiler_pc`, which equals the slot index even after insertions (s256 iter-2 §0). | Every s256 win came from this walk; engine busy tables were useless (nothing above 56%). | Sets the ceilings for #1–#4. Zero gain on its own. | None. | The cost is about 30 minutes. If the path does not pass through the staging reloads, demote #1 and #2. |

If #1–#4 each land at the low end, the total is about 2–3 µs (5%) on a 49 µs kernel. That is above the noise floor but easy to lose in it, so re-measure.

## Closed (from the s256 sibling, not re-tested here; confirm the first one cheaply)
- **Fusing var8 alone**: +0.2%, nothing (s256 iter-1 step 1). Do not test it in isolation.
- **Hoisting a Pool DGE's wait into a preceding `EVENT_SEMAPHORE`**: +0.4 to +0.7 µs, a regression (s256 §8, edit7).
- **Merging the 4 `beta` loads** on Pool (slots 7–11, 14–16.8 µs here): the loads are mergeable, but nothing downstream is gated by them. Their completion was not binding (s256 §6).
- **The long SP `EVENT_SEMAPHORE` blocks** (here, Sync 5.3 µs at 56.2): the extended wait block is the real condition, and they are not on the critical path.
- **Swapping the two tail stores**: net negative on analysis. The final fence waits for both, and the state store is 4× the bytes.
- **Reload-side fusion of var11/var12**: needs 4 dims in a DGE and is impossible. Use the save-side rewrite (#2).
- **The head**: gate-load completion latency is about 3.9 µs at s256 and the slot is already first. No reorder is available.

## Unknown
- **Whether the staging reloads are on this NEFF's critical path.** The critical-path walk (#5) settles it.
- **Whether the Pool DGE gap is real.** Move one Pool store to SP (#4).
- **How large the DMA-latency edges are at s128.** The s256 figures were 2.26 µs for the gate-load row end → semaphore, and 2.9 µs for an SP trigger → queued transfer. A walk on this NEFF will give the s128 values.
- **The margin to the error limit.** Read `baseline/verify_baseline.json`. The fusions are bit-exact in s256, so this should not move.
