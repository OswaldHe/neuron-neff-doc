#!/usr/bin/env python3
"""Decode NEFF debug-info files (sgNN/debug_info_*.dbg) without the protobuf library.

The files are protobuf messages of type `ir_debug_info`; the schema is the one shipped as
neuronxcc/proto/ir_debug_info_pb2.py. Field numbers used here are copied from that schema.
See ../docs/09-debug-info.md.

Usage:
    dbginfo.py <file.dbg>            print the decoded message as JSON
"""

import json
import sys

ENGINE_TYPE = {0: "UNKNOWN_ENGINE", 1: "PE", 2: "ACTIVATION", 3: "POOL", 4: "DMA", 5: "DVE", 6: "SP"}
INSTRUCTION_TYPE = {0: "UNKNOWN_INSTRUCTION_TYPE", 1: "REGULAR", 2: "SPILL", 3: "TRANSPOSE"}
EDGE_KIND = {0: "INVALID", 1: "ORDERED", 2: "ANTI_DEPENDENCE", 3: "OUTPUT_DEPENDENCE",
             4: "FLOW_DEPENDENCE"}


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one message. Length-delimited values are
    returned as bytes; varints as int."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = buf[i:i + ln]
            i += ln
        elif wt == 1:
            v = int.from_bytes(buf[i:i + 8], "little")
            i += 8
        elif wt == 5:
            v = int.from_bytes(buf[i:i + 4], "little")
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fno, wt, v


def _ints(wt, v):
    """A repeated integer field may arrive packed (wire type 2) or one value per tag."""
    if wt != 2:
        return [v]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(x)
    return out


def _instruction(buf):
    ins = {"id": None, "name": "", "parent_ids": [], "dataflow_predecessors": [],
           "scheduling_predecessors": [], "dataflow_predecessor_types": [],
           "scheduling_predecessor_types": [], "instruction_type": None, "attrs": {},
           "stack_frame_id": None}
    for f, wt, v in _fields(buf):
        if f == 1:
            ins["id"] = v
        elif f == 2:
            ins["name"] = v.decode()
        elif f == 3:
            ins["parent_ids"] += _ints(wt, v)
        elif f == 4:
            ins["dataflow_predecessors"] += _ints(wt, v)
        elif f == 5:
            ins["scheduling_predecessors"] += _ints(wt, v)
        elif f == 6:
            ins["instruction_type"] = INSTRUCTION_TYPE.get(v, v)
        elif f == 7:
            k = val = ""
            for f2, _, v2 in _fields(v):
                if f2 == 1:
                    k = v2.decode()
                elif f2 == 2:
                    val = v2.decode()
            ins["attrs"][k] = val
        elif f == 8:
            ins["stack_frame_id"] = v
        elif f == 9:
            ins["dataflow_predecessor_types"] += [EDGE_KIND.get(x, x) for x in _ints(wt, v)]
        elif f == 10:
            ins["scheduling_predecessor_types"] += [EDGE_KIND.get(x, x) for x in _ints(wt, v)]
    return ins


def _stack_frame_index(buf):
    out = {"file_names": [], "function_names": [], "file_locations": [], "stack_frames": []}
    for f, _, v in _fields(buf):
        if f == 1:
            out["file_names"].append(v.decode())
        elif f == 2:
            out["function_names"].append(v.decode())
        elif f == 3:
            loc = {"file_name_id": 0, "function_name_id": 0, "line": 0}
            for f2, _, v2 in _fields(v):
                loc[{1: "file_name_id", 2: "function_name_id", 3: "line"}[f2]] = v2
            out["file_locations"].append(loc)
        elif f == 4:
            fr = {"file_location_id": 0, "parent_frame_id": 0}
            for f2, _, v2 in _fields(v):
                fr[{1: "file_location_id", 2: "parent_frame_id"}[f2]] = v2
            out["stack_frames"].append(fr)
    return out


def decode(data: bytes) -> dict:
    msg = {"description": "", "parent_debug_info_filename": "", "instructions": [],
           "engine_type": None, "kernel_filenames": [], "stack_frame_index": None,
           "kernel_functions": []}
    for f, _, v in _fields(data):
        if f == 1:
            msg["description"] = v.decode()
        elif f == 2:
            msg["parent_debug_info_filename"] = v.decode()
        elif f == 3:
            msg["instructions"].append(_instruction(v))
        elif f == 4:
            msg["engine_type"] = ENGINE_TYPE.get(v, v)
        elif f == 5:
            msg["kernel_filenames"].append(v.decode())
        elif f == 6:
            msg["stack_frame_index"] = _stack_frame_index(v)
        elif f == 7:
            msg["kernel_functions"].append(v.decode())
    return msg


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return decode(f.read())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    print(json.dumps(load(sys.argv[1]), indent=1))
