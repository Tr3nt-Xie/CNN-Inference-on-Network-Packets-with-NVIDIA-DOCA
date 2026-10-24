"""Packet format shared by Part 3 (generator), Part 4 (DPDK receiver) and Part 5.

One 224 x 224 RGB image (150,528 bytes) is too big for one packet, so it is cut into
1,400-byte chunks. Each UDP payload is a 22-byte header followed by one chunk. UDP gives
no ordering, no de-duplication and no notion of which image a datagram belongs to, so
the header carries exactly that:

    offset  size  field        meaning
    0       4     magic        constant that marks our packets
    4       4     frame_id     image this chunk belongs to
    8       2     seq          chunk number within the image, 0 .. nseg-1
    10      2     nseg         total chunks in the image
    12      4     offset       byte offset of the chunk in the image
    16      2     length       image bytes in this packet
    18      4     frame_bytes  total image size

All fields are big-endian (network byte order), like the IP and UDP headers around them.
The C receiver must use the same constants and field order.

File format for images (originals written by gen_pcap.py, rebuilt images written by the
receiver): the images concatenated in frame_id order, FRAME_BYTES each, HWC RGB uint8.
Identical formats mean a receiver can be checked with `cmp originals.bin frames_rx.bin`.
"""
import struct

MAGIC = 0x45453538            # "EE58" in ASCII
CHUNK = 1400                  # image bytes per packet
H, W, C = 224, 224, 3         # ResNet-18 input, RGB
FRAME_BYTES = H * W * C       # 150,528

# Addresses and ports from the handout (Step 3.2)
SRC_MAC, DST_MAC = "02:00:00:00:00:01", "02:00:00:00:00:02"
SRC_IP, DST_IP = "10.0.0.1", "10.0.0.2"
SPORT, DPORT = 40000, 5000

# "!" = network byte order and no padding, so the size is exactly 4+4+2+2+4+2+4 = 22.
HEADER = struct.Struct("!IIHHIHI")
HEADER_BYTES = HEADER.size
assert HEADER_BYTES == 22


def nseg_for(frame_bytes, chunk=CHUNK):
    """Packets needed for one frame: ceiling division without floats."""
    return (frame_bytes + chunk - 1) // chunk
