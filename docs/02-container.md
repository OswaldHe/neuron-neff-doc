# 2. The container: NEFF header and NEFF payload

```
offset 0        NEFF header, 1024 bytes
offset 1024     NEFF payload: gzip stream of a POSIX tar archive
```

## 2.1 NEFF header

All integers are little-endian. Field names are the ones neuron-explorer uses in its
`NeffHeader` table. **Verified**: each field was changed on its own in a copy of a NEFF, and
`neuron-explorer view` reported exactly that field as changed
(`artifacts/experiments/header_bits/`).

| Offset | Size | Field | Value in the corpus | Notes |
|---|---|---|---|---|
| 0x000 | u64 | `package_version` | 2 | |
| 0x008 | u64 | `header_size` | 1024 | Also where the payload starts |
| 0x010 | u64 | `data_size` | payload length | Must equal the payload's byte length |
| 0x018 | u64 | `major_version` | 2 | |
| 0x020 | u64 | `minor_version` | 0 | |
| 0x028 | char[128] | `build_version` | empty | NUL-filled |
| 0x0A8 | u32 | `logical_neuroncore_count` | 1 | |
| 0x0AC | 16 bytes | `hash` | MD5 of the payload | |
| 0x0BC | 16 bytes | (unused) | 0 | |
| 0x0CC | 16 bytes | `uuid` | MD5 of the payload | A profile (NTFF) records the UUID of the NEFF it was captured from; neuron-explorer refuses a NEFF whose UUID differs |
| 0x0DC | char[256] | `network_name` | compile work-dir path of the NEFF | NUL-terminated |
| 0x1DC | u32 | ? | 1 | Not shown by neuron-explorer; meaning unknown |
| 0x1E0 | u32 | ? | 1 | Not shown by neuron-explorer; meaning unknown |
| 0x220 | u32 | feature word | varies | See 2.2 |
| 0x228 | u64 | `pcore_count_per_lnc` | 2 | Physical cores per Logical NeuronCore |
| other | | | 0 | |

`cc_participant_count` (4) and `neuroncore_count`, which neuron-explorer also reports, do not
come from the header. **Verified** (no header byte changes them).

### What the runtime checks

| Header change | Loads and runs? | Evidence |
|---|---|---|
| `hash` zeroed | yes, output bit-exact | experiment r01 |
| `uuid` changed | yes, output bit-exact, profile still captured | experiment r02 |

So the runtime does not compare the payload with the stored MD5. `tools/neff.py pack` still
writes the MD5 into both fields, as the compiler does, so that tools that do check (and
neuron-explorer's NEFF/NTFF matching) keep working. A NEFF you edit gets a new MD5 and therefore
a new UUID; profile it fresh, do not reuse an old NTFF.

## 2.2 Feature word (offset 0x220)

Each bit declares a runtime capability the program needs. Names are neuron-explorer's
(`enabled_features`); each bit was set or cleared on its own and the name it produced recorded.
**Verified.**

| Bit | Name | Bit | Name |
|---|---|---|---|
| 0 | `bin-weights` | 12 | `dge` |
| 1 | `collectives-offset` | 13 | `pool-rsqrt` |
| 2 | `custom-ops` | 14 | `dma-remote-semaphore-increment` |
| 3 | `coalesced-cc` | 15 | `dge-cast` |
| 4 | `queue-set-instances` | 16 | `vector-dge` |
| 5 | `on-chip-functions` | 17 | `embedding-stride32` |
| 6 | `dynamic-pwp` | 18 | `pseudo-embedding-update16` |
| 7 | `imcpy-32b-sem-wait` | 19 | `hw-dge-deprecated` |
| 8 | `imcpy-bound-check` | 20 | `dma-rearm` |
| 9 | `fma-multi-dim` | 21 | `hw-dge` |
| 10 | `virtual-core` | 22 | `partial-cc-sb2sb-support` |
| 11 | `v3-2x-4x-dve-perf-mode` | 28 | printed as `Feature bit` |

Bits 23-27 and 29-31 have no name in this neuron-explorer version.

Bits 10, 11 and 28 are set in every studied NEFF. The others follow what the program uses, for
example bit 3 (`coalesced-cc`) is set exactly in the NEFFs whose definition files list
`neff_coalesced_ccops`, bit 4 exactly where `neff_queue_set_instances` is listed, and bit 12
(`dge`) exactly where a `dynamic` DMA queue exists. **Observed.**

**When editing:** if you add a feature (for example the first DGE instruction in a NEFF that had
none), set its bit too. This is **Inferred**; it was not tested.

## 2.3 NEFF payload

The payload is a gzip stream (`1f 8b 08 00`, MTIME 0, XFL 0, OS 3, deflate level 1, memLevel 8)
of a POSIX `ustar` tar archive:

* every member is a regular file with mode `0000644`, uid/gid 0, mtime 0, empty user and group
  names, NUL-filled `devmajor`/`devminor`, and the checksum written as six octal digits
  followed by two NUL bytes;
* no directory entries (paths like `sg00/def.json` imply the directories);
* the archive ends with exactly two 512-byte zero blocks (no padding to a 10 KiB record).

**Verified**: `tools/neff.py pack` rebuilds all 14 studied NEFFs **byte for byte** from their
unpacked members.

Member order (as written by the compiler): `kelf-0.json`, `neff.json`, `info.json`, then per
subgraph `def.json`, the engine files, the tables and the debug info. `tools/neff.py unpack`
records the order in `members.txt` and `pack` reuses it; new files are appended at the end and
deleted files are dropped. Whether the runtime cares about order was not tested; keeping it
costs nothing.

## 2.4 Unpacking and packing

```bash
python3 neff_doc/tools/neff.py unpack  file.neff  work/      # header.json, header.bin, members.txt, payload/
python3 neff_doc/tools/neff.py pack    work/      new.neff   # recomputes data_size and both MD5 copies
python3 neff_doc/tools/neff.py verify  new.neff               # size and MD5 consistency
```

`pack` keeps every other header byte from `header.bin`. To change a header field, edit
`header.bin` (or the packed file) at the offsets above.
