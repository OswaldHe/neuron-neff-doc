# Review of iter-4 (ssd_mamba2_1b3_s2048)

Compared: `kernel.neff` (working tree) against `HEAD` (iter-3, 5107 us) and against `baseline/` (7154.9 us).
Sources: `neff_diff.py`, `profile_latest.json`, `verify_latest.json`, `.autohelix/notes/iter-4.md`, a fresh
`profile.py --repeats 1 --parquet prof2` (4856.96 us), and `semaphores.py` on unpacked copies.
(`prof/` in the worktree is stale: iter-3 era, 5122 us. Use `prof2/`.)

## 1. What changed, and does it match the notes?

Four members changed against HEAD: `Activation0.bin`, `DVE0.bin`, `PE0.bin` and `SP0.bin`.
Pool0 is untouched. Slot counts are unchanged (ACT 14074, DVE 24525, PE 14865, SP 4061).

| member | slots changed | what |
|---|---|---|
| SP0 | 1024 | The SBUF `dst` of the `state_hbm -> state_sb` DGE (block:238) is repointed to `0x0ec00 + 256*head` in 64 slots. In **960** slots the `PSEUDO_DMA_DIRECT2D` is replaced in place by `EVENT_SEMAPHORE`. That slot copies the DGE's wait (e.g. `sem55 >= 64`) and carries `events_extended_update_mode = 21` (SEM_ADD_IMM_COMPLETE) on the DGE's own semaphore (57-71, etc.) with value 16. |
| ACT0 | 2048 | 1024 in-place `state_sb *= exp_cs_last_N` slots (src and dst) are repointed to `T(head)`. 1024 ACT-queue `state_sb -> state_hbm` DGEs are handled the same way (64 stay as real DMAs). |
| DVE0 | 1024 | The `state_sb += delta` operands (src0 and dst) are repointed. |
| PE0 | 2048 | The two fp32 passes of `C_T @ state_sb` are repointed (moving operand). |

About 408 distinct old addresses collapse onto 64 new ones, `T(h) = 0xec00 + 256*h` (0xec00..0x12b00). The profile agrees:
`spill_reload/save_bytes` 35.6 MB -> 4.2 MB, `hbm_read` -31 MB, `hbm_write` -31.5 MB, SP busy 802 -> 279 us,
DMA busy 1879 -> 1329 us, ACT busy 3672 -> 3510 us. That is 960 x 32 KB x 2 directions = 61 MB.

**This matches the notes and the commit summary.** The notes claim a per-head persistent `state_sb`, 960 loads and
960 stores turned into `EVENT_SEMAPHORE(+16)`, and no wait value moved. I found exactly that: 960 `PSEUDO_DMA_DIRECT2D -> EVENT_SEMAPHORE`
on SP, and the semaphore wait values are untouched.

## 2. Is the latency change real?

Yes.
- This iteration: HEAD 5106.8 -> 4856.0 us (median of 5; min 4854.9, max 4857.3, **spread 0.05%**). That is -250 us (-4.9%), roughly 100 times the spread.
- A separate 1-repeat profile gave 4856.96, and notes record 4856.1.
- Cumulative: 7154.9 -> 4856.0 (-32.1%).
- The win is also corroborated by independent counters (HBM bytes, SP and DMA busy), so it is not noise.

## 3. Safety beyond the verifier

- `verify`: PASS. `max_abs_err` 5.234e-07 and 3.334e-07 against a 6.2e-07 limit (84% and 54%). Notes say this is byte-identical to iter-3.
  Baseline was 5.12e-07 and 3.17e-07. No error was added this iteration, but **84% of the budget is already spent** (iter-1's exp(-x) rewrite),
  so any change that re-rounds an fp32 value is out.
- `semaphores.py` flags sems 40-71 (`max_wait 1088`, `total_inc 128`, "wait exceeds increments"). This is a false alarm:
  the tool does not model `events_extended_update` / `SEM_ADD_IMM_COMPLETE`.
  I confirmed that all 960 replaced SP slots carry ext-update mode 21, value 16, on the DGE's semaphore. Supply is preserved
  exactly (128 real + 960 x 16 = 1088 units), and the device ran to completion.
  Do not "fix" it by lowering waits (see Closed).
- Races: the original ordering of the state round trip is gone, and ordering of a head's chunk c to c+1 now relies on program order and semaphores.
  Slack is large: a head's consecutive uses are ~64 units (~300 us) apart, and every fence is preserved.
  The lifetime argument is sound. The loads and stores that remain are the chunk-0 load, which zeroes `state_sb`, and the chunk-15 store, which feeds `final_state`.
  The risk that survives a passing verifier is a WAR/RAW hazard that the fixed input set cannot expose.
  The `scratch` variant in the notes (loads pointed at a scratch block) is the true control, and it passed.
- One further cheap confidence check I recommend: run `verify.py` 3-5 times, and once with a different input seed if `bench/inputs` can be regenerated.
  Because the output is bit-identical to iter-3, a race would show up as run-to-run variance. Nothing here suggests one.

Verdict: accept. Low residual risk.

## 4. Ranked candidates for iter-5

State of play (`prof2`, 4871 us span): ACT 3510 us busy (72%), PE 1922 (40%), DVE 2439 (50%), Pool 2072 (43%), SP 279 (6%), DMA 1319 (27%).
Wall per unit is ~4.74 us. The engines run a pipeline with ~10 ACTIVATEs per unit and 1.2-1.7 ms of sub-microsecond handoff gaps on ACT and DVE.
Notes §4 measured the **current conversion rate: 184 ns of wall per ACT compute instruction deleted (53% of its busy time)**.
That rate is not stable between iterations, so re-measure it before building anything.
Only bit-exact edits are viable (§6 of the notes: re-rounding an fp32 multiply exhausted the error budget).

| # | candidate | engine / member / slots | change | mechanism and where established | gain / ceiling | risk | cheapest falsifier |
|---|---|---|---|---|---|---|---|
| 1 | **Delete the recomputed `exp_cs_last` ACTIVATE** (block:226, `func=7 nch=1 num_elem=1`), point the 4 `STREAM_SHUFFLE`s at `exp_cs_col` partition 127 | ACT0 (the 1024 nch=1 slots become `EVENT_SEMAPHORE` placeholders) and DVE0 (shuffle src address, `LOAD_MASK_SELECT` mask `[31]*32`, src field 24) | Both are `exp` of the same fp32 number, so deleting is bit-exact. Do the **pairing by data flow** (`exp_cs_last.src_off - 127*4` -> cs_row -> PE transpose -> PSUM -> ACT copy -> `exp_cs_col` whose `src == prod.dst`), not by slot order | `iter-4.md` §7.1: identity and encoding proven (shuffle mask in LMS `immediate_uint8`, 5-bit partition field in units of 4, compiler-confirmed). The only unsolved part is pairing; slot-order pairing gave 5.6e-01 error | 1024 x ~0.2 us x 0.53 = ~100 us ceiling (-2%). Realistic **40-100 us** after the deadlock guard, which costs sites (859 -> ~379 or fewer) | Medium: the wrong pair gives a wrong result (the verifier catches it), and a hang (the verifier catches it). A subtle hazard is the raised sem-4 wait stalling DVE (needs the guard) | The 8-site test with the data-flow pairing: must be **bit-identical** (max_abs_err 5.234e-07). ~2 min. Dies on any other error |
| 2 | **Pool `dt_col` DGE merge**, but only after a destructive probe | Pool0 (1024 DGEs, 4 B/partition, 655 ns each; 671 us of Pool) | Probe first: set `num_elem[0] = 1` on all 3169 Pool DGEs and profile. If wall does not drop, stop. If it does, merge N heads' `dt_col` into one trigger (`BASE + h*4`, `num_elem [128,N]`) | `iter-4.md` §7.2: DGE host cost ~600 ns fixed (other kernels' notes). Pool is only 43% busy with ~2.7-3.0 ms of gaps, so Pool is probably not critical | Unknown; ceiling is the probe's wall drop, and Pool compute is bounded at 2072 -> ~1400 us | Low for the probe (destructive, never shipped). The real edit changes allocations and fences, and the verifier cannot see a missed fence | The probe, ~2 min (`profile.py --repeats 3`): if the wall moves < ~20 us, kill it |
| 3 | **Remove the 64 prologue zero-init ACT DGEs** and the remaining 64 chunk-0 loads and replace them with a DVE memset of `T(h)` | ACT0 (64 DGEs with `src_var_offset == 0x4200 + 256*head`), SP0 (the 64 chunk-0 loads), DVE0 (64 synthesised memsets; the notes show how to build a slot from a zeroed 64 B slot) | `state_hbm` is zero-initialised only so chunk 0 can load zeros. A memset into `T(h)` gives the same bytes. Keep the chunk-15 store and the final-state read-back | Recipe: `SEM_ADD_IMM_COMPLETE(+16)` placeholder (`iter-4.md` §1). The prologue DGEs are identified by the 0x4200 zero tile | 128 DGEs x ~0.55 us = ~70 us ACT/SP busy, mostly in the prologue where latency is dominated by loads. Likely **5-20 us** (-0.1..-0.4%) | Medium: the memset slot is hand-built, and the DGE semaphore supply must be preserved exactly | Probe: replace the 64 prologue DGEs with placeholders without the memset and look at the wall. If < 5 us, drop it |
| 4 | **Does the ACT->DVE rounding gap vanish with `TENSOR_TENSOR` instead of a pointer-scalar op?** | DVE0, the 599 `Y_intra` sites of §6 | Rebuild the fold as a `TENSOR_TENSOR` multiply of `Y_intra_psum` by a stride-0 broadcast `exp_cs_col` instead of `TENSOR_SCALAR_PTR` | `iter-4.md` §6: `TENSOR_SCALAR_PTR` (op 0x44) gave 1.588e-06 error (256% of the limit), 8 sites were clean. The existing `decay_factor` multiply is already a DVE `TENSOR_TENSOR`, so that datapath is already in use | If bit-exact: 599 sites = ~110 us (-2.3%), up to ~1.1 ms for all six multiplies, but DVE (50%) becomes the next constraint after 2-3 moves per unit | High: it is a numerics bet (probability of success unknown), with a hand-built slot and a stride-0 DVE input (unproven) | 8 sites first, then ~100 sites. If error exceeds ~5.3e-07 (iter-3 level) at ~100 sites, stop. ~5 min with `iter-4-fold-NUMERICS.py` |
| 5 | **Software-pipeline the serial per-unit head** (`dt_row -> log_decay -> cs_row -> transpose -> cs_col -> exps -> exp_cs_last -> broadcast -> decay_factor`) | all engines; hundreds of slot moves plus new semaphore chains | Hoist head work for unit n+1 above unit n's tail | `iter-3.md` §4 item 2: ~3 of the 4.74 us per unit is this serial chain | Possibly the largest (hundreds of us), but unmeasured | **Highest.** It reorders cross-engine dependencies; one fixed input set can pass while a race remains | A destructive probe that only drops the serial head's waits to see if the wall moves. Do not start without it |

I would build #1 first (bit-exact, known gain), run #2's probe in parallel because it costs two minutes, and treat #3-#5 as probe-gated.

### Closed (do not redo)

- **Lowering DGE wait thresholds / deleting the DGE slots**: `max_abs_err` 1.47e-01 (delete + remap), 1.28e-01 (loads only). The "scratch dst, no sync change" control passes. Supply must be preserved exactly (SEM_ADD_IMM, iter-4 §2).
- **ACTIVATE merging**: only 42 of 10957 pairs survive the between-window check, and shipping 40 of them gave `max_abs_err` 8.2e-03 (iter-4 §5). Iter-2's 303 pairs were the harvest.
- **Moving any fp32 multiply from ACT to DVE with the pointer-scalar op**: `max_abs_err` 1.588e-06 at 599 sites against the 6.2e-07 limit (iter-4 §6). ACT's rounding is the one that matches torch.
- **Exactly-duplicate ACTIVATE recomputation**: 7343 candidates are different units reusing the same addresses (iter-4 §7.4).
- **"ACT throughput is closed" (iter-2)**: superseded. Conversion is now 53-69% (iter-4 §4). Re-measure it every iteration.
- **Second `exp_cs_last` broadcast**: done in iter-3 (DVE 2701 -> 2440 us, only ~8% of that converted to wall).
- **`exp(x)` reciprocal, activation-table reloads**: done in iter-1 (7155 -> 5130 us).

### Unknown (these would change the ranking)

- Does Pool compute convert to wall at all? Measure it with the destructive `num_elem[0]=1` probe on the 3169 Pool DGEs. This sets or kills #2.
- How much of the 4.74 us per unit is the serial head, as opposed to throughput? Measure it by deleting the head chain's waits (destructive probe) and reading the wall. This sets #5.
- How much error headroom can be recovered? Anything that lowers the exp(-x) rewrite's error would reopen the ACT->DVE family (#4's upside and the 1.1 ms prize). Measure the error of the iter-1 exp(-x) rewrite per output element, to see whether it is a few outliers or systematic.
- Whether the remaining ACT inter-instruction gaps (~1.24 ms of sub-microsecond gaps, ~170 ns each) are semaphore latency or dependency waits. `evt_wait_time_ns` in `prof2/Instruction.parquet` per ACT slot is the measurement. Note that the parquet `duration_ns` on PE sums to more than the span (it includes stalls), so use `engine_busy_us` from `profile_latest.json` for busy time, not the parquet sum.

## Bottom line

This iteration is real (-4.9%, ~100x the spread), consistent with the notes, and safe by construction: the semaphore supply is preserved exactly,
and `semaphores.py`'s alarm is a known false positive. Remaining headroom is modest and mostly gated on ACT instruction count.
The best bit-exact lever is #1 (~40-100 us). Past that, most of the rest is probe-gated or numerically blocked.
