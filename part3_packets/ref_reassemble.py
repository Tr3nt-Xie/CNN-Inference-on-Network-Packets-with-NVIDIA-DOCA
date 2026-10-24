#!/usr/bin/env python3
"""Reference receiver in Python: rebuild images from a PCAP and compare with the originals.

    python3 ref_reassemble.py images.pcap originals.bin [--out frames_ref.bin]

Applies the same rules the DPDK receiver must apply (Part 4): IPv4 + UDP only, destination
port 5000, valid magic and bounds, any arrival order, duplicates ignored. Its output file
uses the same format as originals.bin, so the C receiver can be checked against either.
"""
import argparse
import collections

from scapy.all import IP, UDP, Raw, PcapReader

from frame_format import MAGIC, HEADER, HEADER_BYTES, DPORT, FRAME_BYTES, nseg_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("originals")
    ap.add_argument("--out", default="frames_ref.bin")
    a = ap.parse_args()

    stats = collections.Counter()
    frames = {}                                   # frame_id -> [buffer, set of seqs, nseg]
    for pkt in PcapReader(a.pcap):
        stats["packets"] += 1
        if IP not in pkt or UDP not in pkt:
            stats["drop_not_udp"] += 1
            continue
        if pkt[UDP].dport != DPORT:
            stats["drop_other_port"] += 1
            continue
        payload = bytes(pkt[UDP].payload)
        if len(payload) < HEADER_BYTES:
            stats["drop_short"] += 1
            continue
        magic, fid, seq, nseg, off, length, fbytes = HEADER.unpack_from(payload)
        chunk = payload[HEADER_BYTES:HEADER_BYTES + length]
        if (magic != MAGIC or fbytes != FRAME_BYTES or nseg != nseg_for(fbytes) or seq >= nseg
                or len(chunk) != length or off + length > fbytes):
            stats["drop_bad_header"] += 1
            continue
        f = frames.setdefault(fid, [bytearray(fbytes), set(), nseg])
        if seq in f[1]:
            stats["duplicates"] += 1
            continue
        f[0][off:off + length] = chunk
        f[1].add(seq)
        stats["accepted"] += 1

    complete = sorted(fid for fid, f in frames.items() if len(f[1]) == f[2])
    for fid, f in sorted(frames.items()):
        if len(f[1]) != f[2]:
            missing = sorted(set(range(f[2])) - f[1])
            print(f"frame {fid}: incomplete, missing seq {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    with open(a.out, "wb") as out:
        for fid in complete:
            out.write(frames[fid][0])

    with open(a.originals, "rb") as fo:
        orig = fo.read()
    n_orig = len(orig) // FRAME_BYTES
    match = sum(1 for fid in complete
                if fid < n_orig and frames[fid][0] == orig[fid * FRAME_BYTES:(fid + 1) * FRAME_BYTES])
    print(" ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"complete frames {len(complete)}/{n_orig}, byte-for-byte matches {match}/{n_orig} -> {a.out}")
    raise SystemExit(0 if match == n_orig == len(complete) else 1)


if __name__ == "__main__":
    main()
