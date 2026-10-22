# Review of iter-0 (ssd_mamba2_370m_s2048)

## 1. What changed in the NEFF
Nothing. `neff_diff.py baseline/kernel.neff kernel.neff` reports an identical header, 37/37 identical members and "payloads are identical". `notes/` holds only the template README. There is no `profile_latest.json` or `verify_latest.json`. This iteration produced no edit, so the notes claim nothing and there is nothing to contradict.

## 2. Is there a latency change?
No. I re-profiled the unmodified kernel for a control: 5248.7 us. The baseline is 5248.9 us (min 5248.1, max 5249.3, spread 0.022%). Noise is about 1 us, so any win above roughly 5 us is distinguishable from it.

## 3. Safety
There is no edit to assess. `semaphores.py` on the unpacked baseline reports no wait over its supply. Baseline error is 1.88e-7 against a limit of 2.3e-7 (82%) on output_0, and 52% on output_1. Headroom is thin, so only bit-exact edits are worth trying. Never change arithmetic or accumulation order.

## 4. Where the time is (baseline, 768 units = 48 heads x 16 chunks, about 6.8 us per unit)
Busy times: ACT 4722 us (90%), DVE 1689, Pool 1559, PE 1255, DMA 1193, SP 575.

ACT is the binding engine. Per-opcode ACT time:

| ACT opcode | count | total | mean |
|---|---|---|---|
| ACTIVATE | 7680 | 2769 us | 361 ns |
| **ACT_TABLE_LOAD** | **1531** | **1964 us** (37% of wall) | 1283 ns |
| DMA_DIRECT2D (DGE trigger) | 816 | 430 us | 527 ns |
| EVENT_SEMAPHORE | 1587 | 112 us | |

ACTIVATE breakdown (from `isa.py` on `Activation0.bin`):

- func 128 (copy), pointer imm and pointer scale: 5376.
- func 7 (exp), 1-element, reads PSUM (src marker 2): 1536.
- func 129 (reciprocal): 768.
- Per unit the stream is `f7 exp(PSUM) -> T1 -> f129 recip -> ... -> T0`. That is the same structure as the 1b3 sibling's baseline.

The sibling experiment, `experiments/ssd_mamba2_1b3_s2048_lnc1` (7154.9 -> 4251.0 us, 1.68x), was a different shape of the same kernel. The ranking below is built from its notes, but I checked each premise against this NEFF's own profile and decode.

## 5. Ranked candidates

| # | Edit | Expected gain | Ceiling | Risk | Cheapest falsifier |
|---|---|---|---|---|---|
| 1 | Remove the activation-table reload pair | -1.2 to -1.5 ms | 1964 us table-load time; sibling saw 81% conversion (-2516 us ACT -> -2026 us wall) | medium | One unit |
| 2 | `imm_src` POINTER -> INSTRUCTION on 6144 ACTIVATEs | about -7 to -9% (-350 to -450 us) | 6144 x 91 ns = 559 us ACT busy | low | One ACT stream flip |
| 3 | Delete dead EVENT_SEMAPHORE / LOAD_MASK_SELECT slots | -40 to -50 us | Unknown | low-medium | Count first |
| 4 | Delete the per-chunk `state_sb` HBM round-trip (needs SBUF-hole check) | -4 to -5% (-200 to -250 us) | 6.68 MB spill each way (`summary.spill_*_bytes`) | high | Relayout alone |
| 5 | Merge or shrink ACT work (3-D AP merges, MEMSET shrink) | -50 to -100 us | about 1.6% sensitivity per us of ACT busy once ACT is no longer binding | low | One site |

### 1. Remove the activation-table reload pair
- **Where:** `payload/sg00/Activation0.bin`.
- **Slots:** per unit, about 768 units.
  - the exp (f7, PSUM source)
  - ACTIVATION_TABLE_LOAD sel 1 (T1)
  - the reciprocal (f129)
  - the following ACTIVATION_TABLE_LOAD (T0)
  - Confirm against the slot numbers in my decode (e.g. slots 1570-1574).
- **Change:** use the sibling's iter-1 recipe.
  - Turn the PSUM-reading exp into a func-128 copy of PSUM to the temp buffer.
  - Compute exp(+x) and exp(-x) from that SBUF copy, both in table set 0. The reciprocal of exp(x) is exp(-x).
  - Delete the T0 reload and the T1 load.
  - Keep every semaphore update and wait so no threshold moves. The recipe converted 1020 of 1024 units on the sibling.
- **Mechanism:** table set 0 holds exp, identity and copy but not reciprocal. Set 1 holds reciprocal but not exp. This is documented in `docs/generated/activation_functions.md`, and the sibling's `notes/iter-1.md` §1 and §3 established it on the device.
- **Warning:** do not re-read the same PSUM bank from a second instruction. The sibling hit `PSUM_COLLISION_ERROR` at once, so exactly one PSUM read per bank.
- **Numerics:** exp(-x) is not bit-identical to 1/exp(x). The sibling's error moved from 5.12e-7 to 5.23e-7, a 2% shift. Here the baseline error is 82% of the limit, which leaves room for about 1.9e-8 of movement. The 1-element recip is probably the largest contributor, so check `verify.py` error immediately and bail if it is over 95%.
- **Falsifier:** convert 5 units, run `verify.py`, and look at the error and for any PSUM-collision.

### 2. Flip `imm_src` from POINTER_IMMEDIATE to INSTRUCTION_IMMEDIATE
- **Where:** `Activation0.bin`.
- **Slots:** all ACTIVATEs with `imm_src=POINTER_IMMEDIATE`: 5376 func-128 and 768 func-129, 6144 in total.
- **Mechanism:** the sibling's iter-7 measured this at 91 ns per instruction, 561 us of ACT busy -> 387 us of wall (-8.4%).
- **Premise that does not carry over verbatim:** the sibling's `imm_imm_ptr` was 0. Here `isa.py` decodes `imm_ptr=8421120` (0x808080, a suspicious pattern) and `imm_fp32=1.18e-38`. This may be a decode artefact of the shared field union, which should be checked. The edit is only bit-exact if func 128 and 129 never read imm. Check by diffing a zeroed-imm run against the baseline output.
- **Do not touch** `scale_src=POINTER_IMMEDIATE`. It carries a per-partition scale and is probably used.
- **Falsifier:** flip 100 slots in a single ACT stream, run `verify.py`, and compare to the baseline error. This can run first because it is the cheapest.

### 3. Delete dead slots
- **Slots:**
  - 1579 EVENT_SEMAPHORE on ACT and similar on DVE/Pool/PE/SP with wait NONE and update NONE.
  - `LOAD_MASK_SELECT` (4608 on DVE; remove only those whose next slot is not a STREAM_SHUFFLE).
- **Mechanism:** in the sibling, iter-7 §1 and §2 priced slots by count: -37 us for 3119 slots and -6.7 us for 3077 LMS.
- **Note:** the sibling's dead semaphores were largely leftovers from earlier edits. This baseline may have far fewer, so count them first (grep for `wait=NONE` and `update=NONE`).

### 4. State round-trip removal
- **Mechanism:** the sibling's iter-4 pinned the state tile to a fixed SBUF address and replaced the dead DMA triggers with an EVENT_SEMAPHORE carrying `SEM_ADD_IMM_COMPLETE +16`.
- **Needs:** a free SBUF hole (check `scripts/sbuf.py` or `SbufUsage.parquet`). State per head here is smaller: 32 x 64 fp32 = 8 KB, so about 48 x 8 KB must fit.
- **Risk:** an allocation change that passes a fixed input set can still be wrong. Do the relayout as its own device run first, since the sibling lost several runs here.
- **Gain evidence:** the sibling's rate was -250 us of 5107 (4.9%).

### 5. Merge and shrink small ACT items
- **Mechanism:** `docs` and sibling `notes/iter-6.md` §9 say to shrink a dead instruction rather than replace it with an EVENT_SEMAPHORE. A 3-D access pattern on ACTIVATE worked at 283 sites.
- **Gain:** small, because ACT busy only converts to wall time while ACT is binding. Revisit this after #1.

## 6. Closed (by measurement, from the sibling experiment, so verify before relying on them)
- **Moving DGE triggers off ACT to SP:** +5.2 us worse, and it hangs through `def.json` `dma_queue` (sibling `notes/iter-2.md` §1).
- **Reading PSUM twice:** `PSUM_COLLISION_ERROR` at once (sibling `notes/iter-1.md` §2).
- **Lowering DGE wait thresholds:** `max_abs_err 1.5e-1` (sibling `notes/iter-4.md` §2).
- **A bare `EVENT_SEMAPHORE` replacing a compute slot:** it overtakes the compute and corrupts data unless it carries a wait handshake (sibling `notes/iter-5.md` §1).
- **Pool/DMA reduction:** -193 us of Pool busy and -519 us of DMA busy gave only -12.8 us of wall (sibling `notes/iter-5.md` §5).
- **Pairwise ACTIVATE merging:** nearly exhausted, only 40 of 10957 pairs survived (sibling `notes/iter-4.md` §5).

## 7. Unknown
- **Whether `imm_ptr=0x808080` is real.** Settle it by flipping one stream's slots and running `verify.py`.
- **How much error headroom the exp(-x) rewrite costs on a 1-element recip.** Settle it with the 5-unit trial in #1.
- **Whether ACT is still binding after #1.** After the first edit, re-run the profile: if ACT drops below about 70% busy, drop #3 and #5 and target the DVE/PE ping-pong instead (the sibling measured about 1.4 ms of ACT gap from sem-5/sem-6 handoffs).
- **Free SBUF bytes for #4.** Settle it with `SbufUsage.parquet`.
