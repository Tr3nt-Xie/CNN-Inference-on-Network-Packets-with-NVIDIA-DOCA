#!/usr/bin/env python3
"""Part 3: split test images into UDP packets and write them to a PCAP file.

    python3 gen_pcap.py --images DIR [DIR ...] --out images.pcap --orig originals.bin
                        [--count 16] [--shuffle] [--dup N] [--other-port N] [--seed S]

--images      one or more folders of JPEG/PNG files; images are taken alternately from the
              folders (e.g. cat, dog, cat, ...), in sorted file-name order, until --count
              images have been loaded. Unreadable files are skipped. Without --images,
              synthetic gradient images are used.
--orig        originals in the same format the receiver writes (see frame_format.py),
              plus <orig>.json mapping frame_id -> source file and SHA-256.
--shuffle     shuffle packet order across all images.
--dup N       insert N extra copies of randomly chosen packets.
--other-port N  insert N UDP packets to port 6000, which the receiver must ignore.
The same --seed always produces the same PCAP.
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
from scapy.all import Ether, IP, UDP, Raw, wrpcap

from frame_format import (MAGIC, HEADER, CHUNK, H, W, C, FRAME_BYTES,
                          SRC_MAC, DST_MAC, SRC_IP, DST_IP, SPORT, DPORT, nseg_for)

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def load_image(path):
    """Center-crop to a square, resize to 224 x 224, return HWC uint8 RGB."""
    from PIL import Image
    im = Image.open(path).convert("RGB")      # some files are grayscale or CMYK
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((W, H), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    assert arr.shape == (H, W, C), arr.shape
    return arr


def load_images(folders, count):
    lists = [[os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith(IMAGE_EXT)]
             for d in folders]
    out, idx = [], [0] * len(lists)
    while len(out) < count:
        progressed = False
        for k, files in enumerate(lists):
            while idx[k] < len(files) and len(out) < count:
                path = files[idx[k]]
                idx[k] += 1
                try:
                    out.append((path, load_image(path)))
                    progressed = True
                    break
                except Exception as e:            # corrupt file: skip it
                    print(f"skip {path}: {e}")
        if not progressed:
            raise SystemExit(f"only {len(out)} readable images in {folders}, need {count}")
    return out


def synthetic(count):
    y, x = np.mgrid[0:H, 0:W]
    return [(f"synthetic_{i:02d}",
             np.stack([(x + 13 * i) % 256, (y + 29 * i) % 256, (x + y + 7 * i) % 256], -1)
             .astype(np.uint8)) for i in range(count)]


def frame_packets(frame_id, data):
    nseg = nseg_for(len(data))
    for seq in range(nseg):
        off = seq * CHUNK
        chunk = data[off:off + CHUNK]
        payload = HEADER.pack(MAGIC, frame_id, seq, nseg, off, len(chunk), len(data)) + chunk
        # Lengths and checksums are left to scapy so they are always consistent.
        yield (Ether(src=SRC_MAC, dst=DST_MAC) / IP(src=SRC_IP, dst=DST_IP) /
               UDP(sport=SPORT, dport=DPORT) / Raw(payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--out", default="images.pcap")
    ap.add_argument("--orig", default="originals.bin")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--dup", type=int, default=0)
    ap.add_argument("--other-port", type=int, default=0)
    ap.add_argument("--seed", type=int, default=542)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    imgs = load_images(a.images, a.count) if a.images else synthetic(a.count)
    frames = [img.tobytes() for _, img in imgs]          # HWC, row-major
    pkts = [p for fid, data in enumerate(frames) for p in frame_packets(fid, data)]
    n_data = len(pkts)

    if a.shuffle:
        rng.shuffle(pkts)
    extras = [pkts[rng.randrange(n_data)].copy() for _ in range(a.dup)]
    extras += [Ether(src=SRC_MAC, dst=DST_MAC) / IP(src=SRC_IP, dst=DST_IP) /
               UDP(sport=SPORT, dport=6000) / Raw(b"not for port 5000") for _ in range(a.other_port)]
    for p in extras:                                   # scatter them through the stream
        pkts.insert(rng.randrange(len(pkts) + 1), p)
    # scapy stamps each packet with its creation time; use fixed 1 us steps instead so
    # the same seed always gives a byte-identical PCAP.
    for i, p in enumerate(pkts):
        p.time = 1_800_000_000 + i * 1e-6
    wrpcap(a.out, pkts)

    with open(a.orig, "wb") as f:
        for data in frames:
            f.write(data)
    manifest = {"frame_bytes": FRAME_BYTES, "shape": [H, W, C], "layout": "HWC RGB uint8",
                "frames": [{"frame_id": i, "source": os.path.basename(src) if a.images else src,
                            "sha256": hashlib.sha256(d).hexdigest()}
                           for i, ((src, _), d) in enumerate(zip(imgs, frames))]}
    with open(os.path.splitext(a.orig)[0] + ".json", "w") as f:
        json.dump(manifest, f, indent=1)

    nseg = nseg_for(FRAME_BYTES)
    last = FRAME_BYTES - (nseg - 1) * CHUNK
    ip_len = 20 + 8 + HEADER.size + CHUNK
    print(f"{len(frames)} images x {nseg} packets = {n_data} data packets "
          f"(last chunk {last} B) + {a.dup} duplicates + {a.other_port} other-port "
          f"= {len(pkts)} packets, shuffled={a.shuffle}, seed={a.seed}")
    print(f"largest IP datagram {ip_len} B (MTU 1500, {1500 - ip_len} B to spare), "
          f"Ethernet frame {14 + ip_len} B -> {a.out}, originals -> {a.orig}")


if __name__ == "__main__":
    main()
