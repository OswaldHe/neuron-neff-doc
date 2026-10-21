#!/usr/bin/env python3
"""Just enough NEFF reading for the loop's own checks: the header, and the files inside.

A NEFF is a 1024-byte binary header followed by a gzip-compressed tar archive. This module
reads the header and pulls named members out of the archive. It never writes a NEFF -- for
unpacking, editing and repacking use `neff_doc/tools/neff.py` (chapter 2 and 15).

Usage:
    neffio.py kernel.neff            # header, member list, IO signature
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tarfile

HEADER_BYTES = 1024
PAYLOAD_OFFSET = 0x008          # u64: where the gzip payload starts
PAYLOAD_SIZE = 0x010            # u64: how many bytes of payload
MD5_OFFSET = 0x0AC              # 16 bytes: md5 of the payload (a second copy sits at 0x0CC)


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES:
        raise ValueError(f"{path}: shorter than a NEFF header")
    off = struct.unpack_from("<Q", head, PAYLOAD_OFFSET)[0]
    size = struct.unpack_from("<Q", head, PAYLOAD_SIZE)[0]
    return {
        "format_version": struct.unpack_from("<Q", head, 0)[0],
        "payload_offset": off,
        "payload_size": size,
        "payload_md5": head[MD5_OFFSET:MD5_OFFSET + 16].hex(),
        "file_bytes": os.path.getsize(path),
    }


def members(path: str) -> dict[str, bytes]:
    """{member name: bytes} for every file inside the NEFF's payload."""
    h = read_header(path)
    with open(path, "rb") as f:
        f.seek(h["payload_offset"])
        payload = f.read(h["payload_size"])
    out = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r:") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read()
    return out


def header_problems(path: str) -> list[str]:
    """Empty if the header agrees with the payload: right size, right MD5 in both copies.

    `neff_doc/tools/neff.py pack` recomputes these; a hand-patched file that skipped the repack
    shows up here instead of as a strange runtime failure.
    """
    h = read_header(path)
    problems = []
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        f.seek(h["payload_offset"])
        payload = f.read()
    if len(payload) != h["payload_size"]:
        problems.append(f"payload is {len(payload)} bytes, the header says {h['payload_size']}")
    md5 = hashlib.md5(payload[:h["payload_size"]]).hexdigest()
    for off in (MD5_OFFSET, 0x0CC):
        got = head[off:off + 16].hex()
        if got != md5:
            problems.append(f"md5 at {off:#x} is {got}, the payload hashes to {md5}")
    try:
        members(path)
    except Exception as exc:                                         # noqa: BLE001
        problems.append(f"payload does not unpack: {type(exc).__name__}: {exc}")
    return problems


def io_signature(path: str) -> dict:
    """The NEFF's interface: input tensor names in order, and how many outputs it has.

    Read from `neff.json`: the graph's `arg_nodes` are the inputs and `heads` the outputs.
    `info.json` says how many NeuronCores (TPBs) the NEFF needs -- 1 for an LNC=1 NEFF.
    """
    m = members(path)
    graph = json.loads(m["neff.json"])
    nodes = graph["nodes"]
    info = json.loads(m["info.json"]) if "info.json" in m else {}
    return {
        "inputs": [nodes[i]["name"] for i in graph["arg_nodes"]],
        "n_outputs": len(graph["heads"]),
        "num_tpb": info.get("num_tpb"),
        "subgraphs": sorted({n.split("/")[0] for n in m if "/" in n}),
        "members": len(m),
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "kernel.neff"
    print(f"{p}  sha256 {sha256(p)}")
    print(json.dumps(read_header(p), indent=2))
    print(json.dumps(io_signature(p), indent=2))
