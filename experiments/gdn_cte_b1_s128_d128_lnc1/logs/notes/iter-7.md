# iter-7 — **the two tail output stores are no longer DGE rows.** A new `"type": "in"` DMA queue
# can be *added* to `def.json`, and a queued `PSEUDO_DMATRIGGER` on it can write an **IO
# (output) variable**, which the spill/reload data queue flatly refuses. That replaces
# `row 667 ns + 650 ns to first packet` with `trigger 65 ns + 750 ns`, i.e. **−0.5 us on the
# metric (−1.2 to −1.3%)**, the largest single win since iter-4.

## Result: anchor (iter-6 bytes) **38.07** -> shipped **37.54**. Fourteen 7-capture runs and
## **three order-balanced ABBA blocks**:
## * anchor: 38.053 / 38.098 / 38.022 / 38.092 / 38.092
## * E2 (one new queue): 37.572 / 37.568 / 37.635 / 37.598 / 37.594
## * E5 (shipped): 37.468 / 37.476 / 37.528 / 37.644 / 37.592 / 37.633 (last, final bytes)
## * ABBA #1 anchor/E2/E2/anchor -> A 38.060, B 37.602, **Δ −0.458**
## * ABBA #2 E2/E5/E5/E2 -> A 37.596, B 37.502, **Δ −0.094** (every E5 run < every E2 run)
## * ABBA #3 anchor/E5/E5/anchor -> A 38.092, B 37.618, **Δ −0.474**
## The device drifted *slower* over the session (E5 read 37.47 early and 37.59-37.64 late, the
## anchor 38.02 early and 38.09 late), so **quote the ABBA deltas: 38.07 -> 37.54, −0.5 us
## (−1.3%)**, not the raw minima. `verify.py --runs 3` PASS, 3/3 bit-identical,
## `max_abs_err` identical to the baseline to every digit (output_0 6.119e-05 = 17%,
## output_1 2.864e-04 = 82%). `launch_rules.py` PASS. md5 `3a1b0c3e17b76e5ae724723441c0b95a`.

| step | edit | verdict |
|---|---|---|
| E1 | output_1 store as a queued transfer on the existing **data** queue `qSPSpillReload0` | **REFUTED at load time** — `TDRV:drs_expand_data_desc_model Var 7 could not be found. Make sure the variable exists and is not an IO variable. Failed to process descriptor block 21 for queue qSPSpillReload0_0` |
| E1b | same, but on a **new `qSPIO0`, `"type": "in"`** queue added to `def.json` | PASS, **38.05 -> 37.861** |
| **E2** | + the sem-16 output_0-hi store (`ACT[81]` DGE row) also queued, same queue | PASS, **37.57** (it became the binder after E1b) |
| E3 | + output_1's 64 kB split over **two rings of one queue** (ids 0 and 2) | **REFUTED, 37.78 (+0.2)**: rings of one queue share the descriptor dispatcher |
| E4 | E2 but output_0-hi on a **second new queue `qSPIO1`** | 37.623 — a wash vs E2 (faster dispatch, colder start, §3) |
| **E5** | E4 + a 4-byte **warm-up** transfer on `qSPIO0` gated `S[61]>=1` | **SHIPPED**, −0.09 vs E2 in ABBA (mechanism is *not* the one it was built on, §4) |
| E6 | E5 + output_1's 64 kB split over **two queues** (`qSPIO0` + a third queue `qSPIO2`) | **REFUTED, 37.83 (+0.3)**: both halves' first packets came at doorbell **+840** instead of +750 and the two halves ran almost serially (37.63-38.18 and 37.81-38.21) — **the descriptor *fetch* is the serialized resource, not the dispatcher** (§3) |

Shipped diff vs iter-6: `sg00/{def.json, SP0.json, SP0.bin 41->43 slots, Activation0.bin, Pool0.bin}`.

---

## 1. THE FINDING: you can add a DMA queue, and an `"in"` queue may write an output

* A **queued** transfer is cheaper than a **DGE row** by ~0.55 us at the end of the kernel:
  * DGE row: `gate -> +0.02 -> row 667 ns -> +650 ns -> first packet -> span`
  * queued:  `gate -> +0.02 -> trigger 65 ns (2 WRITEs, 130 ns for a cold queue) -> +720..750 -> first packet -> span`
* **The data queue (`"type": "data"`) rejects IO variables at load time** (error above, `nrt_load`
  fails with `NRT_INVALID`, no hang, no device risk). The compiler never puts an output store on
  it; in the corpus (`HCSublayerIn/sg01`) outputs are stored by `qSPIO0`, `"type": "in"`.
* **Adding a queue works.** `def.json -> dma_queue`:
  ```json
  "qSPIO0": {"type": "in", "num_queues": 16, "owner": "sp", "fabric_path": "main",
             "semaphore_set": [18,19,20,21,22,23,40,41,42,54,55,62,63,7,1,0]}
  ```
  Only `semaphore_set[id % 16]` of the ids you actually use is ever signalled, so the other 15
  entries can be any ids (I filled them with the remaining free ones). Two such queues
  (`qSPIO0`, `qSPIO1`, both `owner: sp`) also load and run.
* The transfer itself goes in `SP0.json -> dma` with its own id space per queue (`id` 0,1,2...),
  `"semaphore": semaphore_set[id % 16]`, and a plain descriptor — no `transpose_*` keys:
  ```json
  {"op": "copy", "num_tiling_dimensions": 1,
   "from": "SB", "from_dtype": "float32", "from_off": 18192,
   "from_sizes": [512, 128], "from_steps": [1, 262144],
   "to": "output_1", "to_dtype": "float32", "to_off": 0,
   "to_sizes": [512, 128], "to_steps": [1, 512]}
  ```
  `"op": "cast"` with `from_dtype float32 / to_dtype bfloat16` reproduces a DGE row's
  `in_dtype/out_dtype` conversion exactly (`from_sizes[0]` is the *source* byte count,
  `to_sizes[0]` the destination's: 512 -> 256). Both forms were bit-exact on the device.
* The trigger is a `PSEUDO_DMATRIGGER` **on Sync only** (every one of the corpus's 45,909 is on
  SP). Build it by copying an existing trigger slot and setting `count_union_block_id`,
  `events_wait_idx/_semaphore_value` and the 32-byte NUL-padded `dma_queue_name`. Inserting
  slots at the end of `SP0.bin` (41 -> 43) is fine.
* Re-point the exit fence: `Pool[22]` is `ES{W S[17]>=16, XW S[16]>=16}` (the two DGE stores'
  semaphores) -> `W S[18]>=16, XW S[20]>=16` (the two queue semaphores). A queued transfer adds
  `num_queues = 16` on completion, so the threshold stays 16 for the first transfer on a
  semaphore and 32 for the second. `Pool[23]` is a `NOP{W S[15]>=16}` (the third store, still a
  DGE row on Pool) and its struct has **no extended-wait field**; `Pool[20]`/`Pool[21]` are
  wait-less `EVENT_SEMAPHORE`s and are free slots for an extra fence (Pool has nothing to do
  between 33.3 and the epilogue).

## 1a. The exact shipped delta (so it can be rebuilt or trimmed without re-deriving it)

```
def.json  dma_queue += qSPIO0 {type in, num_queues 16, owner sp, fabric_path main,
                               semaphore_set [18,19,20,21,22,23,40,41,42,54,55,62,63,7,1,0]}
                    += qSPIO1 {same, semaphore_set [20,21,22,23,40,41,42,54,55,62,63,7,1,0,18,19]}
SP0.json  dma    += {id 0, qSPIO0, sem 18, copy  SB p0+0x4710  [512,128] step[1,262144]
                                           -> output_1 off 0      [512,128] step[1,512]  fp32}
                 += {id 1, qSPIO0, sem 19, copy  SB p0+0x37900 [4,1]     step[1,262144]
                                           -> SB   p0+0x37900 [4,1]      step[1,262144]  (warm-up)}
                 += {id 0, qSPIO1, sem 20, cast  SB p64+0x4000 [512,64]  step[1,262144]
                                           -> output_0 off 0x4000 [256,64] step[1,256]  fp32->bf16}
SP0.bin   slot 40 (was the sem-17 DGE row) -> DMATRIGGER blk1 qSPIO0, W S[61]>=1
          slot 41 (new)                    -> DMATRIGGER blk0 qSPIO1, W S[6]>=65
          slot 42 (new)                    -> DMATRIGGER blk0 qSPIO0, W S[6]>=66
Act0.bin  slot 81 (was the sem-16 DGE row) -> EVENT_SEMAPHORE, no wait, no update
Pool0.bin slot 22 exit fence  W S[17]>=16 -> W S[18]>=16 ; XW S[16]>=16 -> XW S[20]>=16
```
To fall back to E2 (one queue, no dummy, −0.46 instead of −0.50, simpler): drop `qSPIO1` and the
warm-up, put the output_0-hi store on `qSPIO0` as id 1 with sem 19, keep two triggers
(`W S[6]>=65` then `W S[6]>=66`), and make `Pool[22]`'s extended wait `S[19]>=16`.

## 2. The three output stores of this kernel (addresses are stable, slot indices are iter-7's)

| store | was | writes | source | gate |
|---|---|---|---|---|
| output_0 lo | `Pool[18]` DGE row, sem 15, 686 ns | `output_0 + 0`, 64x256 B bf16 | `p0..63 + 0x37900` fp32 | `S[6]>=58` (~32.3 us) |
| output_0 hi | `ACT[81]` DGE row, sem 16 -> **queued blk0 on qSPIO1** | `output_0 + 0x4000` | `p64..127 + 0x4000`, written by `DVE[99]` | `S[6]>=65` |
| output_1 | `SP[40]` DGE row, sem 17, 667 ns -> **queued blk0 on qSPIO0** | `output_1 + 0`, 128x512 B fp32 (64 kB) | `p0..127 + 0x4710`, written **entirely** by `DVE[100]` | `S[6]>=66` |

`DVE[100]` writes all 128 partitions x 512 B of the output_1 store's source, so **splitting that
store to start part of it earlier is impossible** — there is no half that is ready sooner.

## 3. The DMA numbers that decide every store edit (all measured this iteration)

* **A queue's descriptor dispatcher issues one packet every ~24 ns when the queue carries only
  that transfer** (4100-byte packets -> 178 GB/s, the same as the DGE path's 182 GB/s), **~32 ns
  when a second transfer shares the queue**, **~35 ns when the same transfer is split over two
  rings of one queue**. Rings of one queue therefore **share one dispatcher** (E3, +0.2).
  Two *queues* each get their own (E4: output_1 dispatched at 24 ns while output_0-hi ran).
* **The largest packet the descriptor generator emits is 4096 B + a 4-byte semaphore write**
  (= the 4100-byte packets). It coalesces 8 partition-strided 512-B source runs into one packet
  because the *destination* is contiguous. A `cast` descriptor does **not** coalesce: the 16 kB
  output_0-hi store goes out as 64 packets of 256 B (~5.6 ns apart, 45 GB/s). So for a queued
  store, **fewer/bigger packets win** — do not break coalescing.
* **Queued completion semaphore writes are packets too** and are merged into the data packets
  (4100 = 4096 + 4); a DGE row instead emits a separate 4-byte packet per channel. Both end up
  inside `max(DmaPacket.end_ts)`, i.e. **inside the metric**.
* Trigger -> first packet: **720-750 ns cold**, 519 ns observed once (E2, when a 16 kB transfer's
  doorbell on the same queue had been rung 327 ns earlier). A 4-byte dummy 734 ns earlier did
  **not** reproduce it (§4).
* **The descriptor fetch is a single serialized resource across queues** (E6): with the 64 kB
  store split over two queues, *both* halves' first packets slipped to doorbell + 840 and the two
  halves' packet trains barely overlapped. So the ~0.75 us fetch cannot be parallelised and
  **more transfers in the tail is a cost, not a win** — the 0.57 us span is already the cheap part.
* A `PSEUDO_DMATRIGGER` expands to **two** `WRITE`s on Sync (65 + 68 ns) for a cold queue, one
  (65 ns) when the queue was used before.
* The tail store budget now: `DVE[100] end -> +0.03 trigger -> +0.75 fetch -> 0.57 span (16x4 kB
  at 24 ns + 180 ns last packet) = +1.24..1.27 us`, against `+1.70` for the DGE row.
  **The floor is ~1.15** (0.5 fetch + 0.65 for 80 kB at 170 GB/s); there is ~0.1 left, no more.

## 4. E5 ships but its stated mechanism is refuted — read this before building on it

E5 adds a 4-byte SB->SB self-copy on `qSPIO0` (id 1, sem 19) triggered at `S[61]>=1`, ~0.73 us
before the real store's doorbell, on the theory that a descriptor fetch already in flight makes
the next one cost 126 ns instead of 750. **On the device the real store's first packet still came
at doorbell + 750**, so the theory is wrong, yet E5 measures 0.09 lower than E2 in an ABBA block
and in every single run. Two candidate reasons, neither confirmed:
* in E5 the output_0-hi store is on its own queue and finishes 0.5 us before output_1, so
  output_1 is the only binder and gets the 24 ns dispatch rate;
* the dummy's own 16 completion packets are dispatched *after* everything else (they ended 6 ns
  after output_1's last packet in the capture I analysed) — **a structural hazard: in a worse
  capture they could become the metric's end.** If a future iteration sees the tail regress,
  drop the dummy (that is exactly `work_save/e4.neff` = 37.62) or move it to `qSPIO1`.
* The **chain-relative** measure (`DVE[100] end -> last packet`) is 1240 ns for E2, 1262 for E4,
  1265 for E5 — i.e. identical. The metric difference comes from somewhere else; capture drift
  is ±0.3 us, so **only run-count and strict separation support E5 over E2**, not a mechanism.

## 5. Where the remaining 37.5 us is (critical path re-derived from this iteration's profile)

Backward walk (`scripts/cp.py`, gate = latest semaphore landing, else the previous op on the
engine). Every link has 23-59 ns of slack; nothing is engine-blocked:

```
blk16 (S[24]#32) -> PE[128/129] -> S[5]#58 -> DVE[88] -> S[6]..-> ACT[71] -> S[4]#34
  -> PE[131..134] -> S[5]#60 -> DVE[90] -> DVE[91] -> DVE[92] -> PE[136/137]
  -> blk18 (S[26]#32) -> PE[138/139] -> S[5]#61 -> DVE[93] -> S[6]#60
  -> Pool[19] AFFINE_SELECT 235 ns -> S[3]#7 -> DVE[95] CAST 192 ns -> S[6]#62
  -> blk19 TRIGGER -> (1.6 us hop) -> S[27]#32 -> PE[142/143] -> S[5]#63 -> DVE[98] 283 ns
  -> S[61] -> DVE[99] 285 ns ----\
  (blk20 arm: S[4]#39 -> blk20 TRIGGER -> 1.3 us hop -> S[28]#32 -> ACT[80] 389 ns
   -> S[4]#41 -> PE[144/145] 289 ns ----)
  -> max(DVE[99] end, PE[145] end)  ** the two arms now arrive within 2 ns **
  -> DVE[100] 292 ns -> S[6]#66 -> output_1 trigger -> +0.75 -> 0.57 span -> METRIC END
```

* **Four queued DMA hops (blk16, blk18, blk19, blk20) at 1.3-1.6 us each are now the only big
  items left**, and the last two sit on the *two different arms of a diamond that is balanced to
  2 ns*, so shortening one arm is worth 0. blk19 is a 64x64 bf16 DMA transpose feeding
  `PE[142]`'s LDWEIGHTS; blk20 is the 256-byte 1->64 partition scatter feeding `ACT[80]`'s
  per-partition scale pointer (iter-6 §7.1). Killing *both* (PE identity-matmul transposes into
  PSUM + an evacuation op each) is the only remaining lever with a >0.5 us ceiling, and it is new
  arithmetic in PSUM with 18% of the error budget left on output_1.
* `Pool[18]`'s DGE row (686 ns) ends **8 ns** before `Pool[19]`'s gate lands. `Pool[19]` is on the
  binding blk19 arm, so on an unlucky capture that row delays the whole arm. **Converting the
  output_0-lo store to a queued transfer (`qSPIO1` id 1 -> sem 21, trigger on Sync gated
  `S[6]>=58`) removes the row from Pool entirely** — the cheapest remaining edit, worth 0 to 0.07
  plus a variance reduction. I ran out of time to measure it.
* The head/body before blk16 was closed in iter-5/6 (head gains do not reach the exit; blk14's
  and blk18's triggers are disguised WAR guards).

## 6. Do NOT retry (measured here, on top of iters 3-6)

* An output store on the **data** queue: load-time error (§1).
* **Splitting a queued store over two rings of one queue** (E3, +0.2) and splitting a DGE store
  (iter-3 §5, iter-4 §8) — both dead for the same reason: one dispatcher, fixed per-packet cost.
* **Two queues instead of one for the two stores** (E4): a wash on its own (it is in the shipped
  bytes only because E5 was built on top of it).
* **Splitting a queued store over two queues** (E6, +0.3) — the fetch serialises (§3).
* More generally: **adding transfers to the tail costs ~0.1 us each in fetch time.** The next
  store edit should *remove* a transfer, not add one.
* Breaking packet coalescing to get 512-B packets out of a queued store: the queued dispatcher
  costs ~5.6 ns per small packet (45 GB/s) vs 24 ns per 4 kB packet (178 GB/s) — small packets
  are *worse* here, unlike on the DGE path.
* Swapping `DVE[99]`/`DVE[100]` (their operands are disjoint, so it is legal, and the two S[6]
  counts would just trade): worth ~0, `DVE[100]` already starts 38 ns after its gate `S[5]#64`
  and `DVE[99]` runs underneath `PE[145]` for free.
* Lowering a queued block's wait below its 16th increment (iter-6 §2) — still a real race.

## 7. Ranked candidates for iter-8

| # | candidate | ceiling | risk | cheapest falsifier |
|---|---|---|---|---|
| 1 | **Kill both tail DMA hops** — blk19 (64x64 bf16 transpose for `PE[142]`'s LDWEIGHTS) and blk20 (256 B 1->64 scatter for `ACT[80]`'s scale pointer), ~1.3-1.6 us each, on the two arms of a diamond balanced to 2 ns, so **both or nothing** | 1.0-2.0 us | high: new PE/PSUM instructions, new semaphores, output_1 has 18% of the error budget left | does a free PSUM bank and an identity/one-hot tile already exist? (`scan.py` over the AFFINE_SELECT outputs) |
| 2 | The remaining two body hops, **blk16** (`S[24]#32 -> PE[128]`) and **blk18** (`S[26]#32 -> PE[138]`), same shape | 1.0 us | high | their triggers are already gate-bound and blk18's is a disguised WAR (iter-5 §1) — only replacing the transfer helps, not moving it |
| 3 | `Pool[18]` (output_0-lo DGE row, 686 ns) -> a queued transfer on `qSPIO1` id 1 (sem 21), trigger on Sync gated `S[6]>=58`; `Pool[23]`'s `W S[15]>=16` -> `W S[21]>=16` | 0-0.07 + removes the 8 ns margin by which that row can delay `Pool[19]`, which gates the binding blk19 arm | low | profile 7 captures: `Pool[19]`'s start minus its gate should stay 59 ns in *every* capture |
| 4 | Drop E5's warm-up dummy if the tail ever regresses (§4) | — | — | compare against the E2 recipe in §1a |
| 5 | `Pool[19]` (235 ns AFFINE_SELECT) + `DVE[95]` (192 ns CAST) sit between `S[6]#60` and blk19's trigger and are strictly serial; iter-6 refuted moving the CAST, but **moving the AFFINE_SELECT** was never tried (Pool is its only host in this NEFF, so it needs a different opcode on Activation/Vector) | 0.2 | high | is there any `TENSOR_SCALAR_AFFINE_SELECT` on a non-Pool engine anywhere in the corpus? |

## 8. Method / tools (`scripts/` is not preserved; ~20 min to rebuild)

* `scripts/lib.py` — load/save a stream as 64-byte slots, `brief()` per slot (waits, extended
  waits, updates, DGE `src/dst` as `var<id>+off` or `tbl<idx>+off`, trigger queue name + block
  id). `isa.Instruction` has `.name` (not `opcode_name`), `.as_dict()`, `.set()`, `.encode()`.
* `scripts/cp.py <profdir> <ENGINE> <slot>` — the backward critical-path walk above. It needs
  `SemaphoreUpdate.id` parsed with `S\[(\d+)\]` (the column is a *string* like `S[6] (Vector)`).
* `scripts/abba.sh a b b a` — copies `work_save/<name>.neff` over `kernel.neff` and runs one
  7-capture profile each. **Two ABBA blocks resolved 0.09 us; plain medians could not.**
  > **TRAP, nearly lost this iteration: an ABBA block ends with the *anchor* in `kernel.neff`.**
  > I ran `abba.sh anchor e5 e5 anchor` as the last device step and `kernel.neff` was left holding
  > the iter-6 bytes — `verify.py` and `launch_rules.py` both still PASS, so nothing complains.
  > **`md5sum kernel.neff` against the config you mean to ship after every A/B run**, or make the
  > script restore the shipped file at the end.
* **Ingest any repeat, not just the last:** `tools/profile.py` keeps all NTFFs in `.profile/`;
  `python -c "import sys;sys.path.insert(0,'tools');import device as D;
  D.view_parquet('.profile/r1_exec_5.ntff','prof_x')"`. The `--parquet` dir is the *last* capture,
  which was the worst repeat twice this session — always check which capture you are reading.
* `DmaPacketAggregated` gives `block_id, queue_name, op, variable, transfer_bytes, trigger,
  start_ts, end_ts` — one line per transfer, and it **names the output variable**, which is how
  the trigger -> transfer mapping was confirmed. `DmaPacket` gives per-packet `engine_idx`
  (E64..E79 = the 16 channels) and is what the metric's end is taken from.
* Device cost: 6 verifies (1 load-time failure, 0 wrong answers), 17 profiles, no hang.
* **Free semaphores after this iteration: 21, 22, 23, 40, 41, 42, 54, 55, 62, 63, 7, 0, 1**
  (18 = qSPIO0's output_1 store, 19 = the warm-up dummy, 20 = qSPIO1's output_0-hi store;
  15 still = the Pool output_0-lo DGE row; 16 and 17 are now **unused**; **2 is the runtime
  barrier — do not touch**).
* `semaphores.py` is still blind to extended waits (sem 19/20 print `max_wait 0`) — as always,
  `verify.py --runs 2` before every profile.
