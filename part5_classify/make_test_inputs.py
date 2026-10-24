#!/usr/bin/env python3
"""Make synthetic fixtures for Part 5 development; these are NOT DPDK results."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a new or empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(542)
    frames = rng.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    order = rng.permutation(16)
    np.save(args.out / 'original.npy', frames)
    (args.out / 'ordered.bin').write_bytes(frames.tobytes())
    np.savez(args.out / 'shuffled.npz', frames=frames[order], frame_ids=order)
    (args.out / 'FIXTURE_ONLY.json').write_text(json.dumps({
        'purpose': 'Synthetic Part 5 smoke test only; not generated or reconstructed by Parts 3 and 4',
        'seed': 542, 'frames': 16, 'shuffled_frame_ids': order.tolist(),
    }, indent=2) + '\n', encoding='utf-8')
    print(f'Synthetic fixtures written to {args.out}; do not report them as DPDK reassembly evidence.')


if __name__ == '__main__':
    main()
