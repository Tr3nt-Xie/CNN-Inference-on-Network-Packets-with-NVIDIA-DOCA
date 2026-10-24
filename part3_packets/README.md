# Part 3: Images as Network Packets

Owner: Fengmao Xie

PDF Steps 3.1-3.2: a packet header, a Python/Scapy generator that writes 16 test images
to a PCAP (in order and shuffled), and the saved original images.

## Files

| File | Purpose |
|---|---|
| `frame_format.py` | The packet format and constants shared with Parts 4 and 5 (the C receiver mirrors it) |
| `gen_pcap.py` | Loads images, splits them into UDP packets, writes a PCAP and the originals |
| `ref_reassemble.py` | Reference receiver in Python: rebuilds images from a PCAP and compares them with the originals |
| `run_part3.sh` | Builds the three PCAPs below and checks each with the reference receiver |

## Packet format (Step 3.1)

Each UDP payload is a 22-byte header followed by one chunk of image data. All fields are
big-endian (network byte order).

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | `magic` | `0x45453538` ("EE58"), marks our packets |
| 4 | 4 | `frame_id` | image this chunk belongs to (0-15) |
| 8 | 2 | `seq` | chunk number within the image, 0 .. nseg-1 |
| 10 | 2 | `nseg` | total chunks in the image |
| 12 | 4 | `offset` | byte offset of the chunk in the image |
| 16 | 2 | `length` | image bytes in this packet |
| 18 | 4 | `frame_bytes` | total image size (150,528) |

Addresses follow the handout: `02:00:00:00:00:01 -> 02:00:00:00:00:02`,
`10.0.0.1 -> 10.0.0.2`, UDP `40000 -> 5000`.

**Packets per image.** 150,528 / 1,400 = 107.52, so each image needs **108 packets**; the
last carries 728 bytes. The largest IP datagram is 20 (IP) + 8 (UDP) + 22 (header) + 1,400
= **1,450 bytes**, 50 bytes under the 1,500-byte MTU (1,464-byte Ethernet frame).

## Images

16 images, 224 x 224 RGB, taken alternately from the cat and dog test folders of the
Lab 7 dataset (Hugging Face `microsoft/cats_vs_dogs`), each center-cropped to a square and
resized. The dataset images are not committed; pass their folders on the command line.
Without arguments the script uses synthetic gradient images.

## Run

```bash
bash part3_packets/run_part3.sh /path/to/test/cat /path/to/test/dog
```

Outputs in `data/part3/` (not committed):

| File | Contents |
|---|---|
| `images.pcap` | 1,728 packets in order |
| `images_shuffled.pcap` | the same 1,728 packets, shuffled across all images |
| `images_stress.pcap` | shuffled, plus 50 duplicate packets and 20 packets to UDP port 6000 (1,798 total) |
| `original.bin` | the 16 originals, concatenated in frame_id order, 150,528 bytes each, HWC RGB uint8, no headers |
| `original.json` | frame_id -> source file name and SHA-256 |

The seed is fixed (542) and packet timestamps are fixed, so the same images always give
byte-identical PCAPs.

## Contract with Parts 4 and 5

- **Part 4** writes the rebuilt images in the same layout as `original.bin` (frame_id
  order, no headers), so its output can be checked with
  `cmp data/part3/original.bin frames_rx.bin`.
- **Part 5** reads both files directly: `part5_classify/frame_io.py` accepts a headerless
  `.bin` and numbers frames 0..N-1 in file order. Checked: `load_frames` on
  `original.bin` returns 16 frames, shape (16, 224, 224, 3), IDs 0-15.

## Results

Run log: [`results/part3/run-20261024T194641Z.log`](../results/part3/run-20261024T194641Z.log)
(Docker, Ubuntu 22.04 with the same apt DPDK and pip Scapy as the instance).

| PCAP | Packets | Accepted | Duplicates dropped | Other-port dropped | Rebuilt byte for byte |
|---|---:|---:|---:|---:|---:|
| in order | 1,728 | 1,728 | 0 | 0 | 16/16 |
| shuffled | 1,728 | 1,728 | 0 | 0 | 16/16 |
| shuffled + stress | 1,798 | 1,728 | 50 | 20 | 16/16 |

These are the Python reference receiver's results; the DPDK receiver is Part 4.
