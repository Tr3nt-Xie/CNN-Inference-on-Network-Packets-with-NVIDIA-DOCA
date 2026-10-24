# Part 4: Receiving Packets with DPDK

Owner: Fengmao Xie

PDF Steps 4.1-4.2: `dpdk_rx.c` reads a Part 3 PCAP through the DPDK pcap driver, checks
the Ethernet/IPv4/UDP headers, rebuilds each image in any arrival order, ignores duplicate
chunks, and writes the complete images to a file for Part 5.

## Files

| File | Purpose |
|---|---|
| `dpdk_rx.c` | The receiver |
| `Makefile` | `make` builds `dpdk_rx` against the system DPDK via `pkg-config libdpdk` |
| `run_part4.sh` | Builds the receiver and runs it on every Part 3 PCAP plus a packet-loss case; logs to `results/part4/` |

## Run

After `bash part3_packets/run_part3.sh ...` has written `data/part3/`:

```bash
bash part4_dpdk/run_part4.sh
```

One PCAP by hand (handout Step 4.2):

```bash
make -C part4_dpdk
sudo ./part4_dpdk/dpdk_rx -l 0 --no-huge -m 512 --no-pci \
  --vdev 'net_pcap0,rx_pcap=data/part3/images_shuffled.pcap,tx_pcap=/dev/null' \
  -- data/part4/frames_rx.bin [--expect 16] [--idle-ms 200]
```

Arguments after `--` go to the program: `rte_eal_init` returns how many arguments it used,
and the program shifts `argc`/`argv` by that amount so its own first argument is `argv[1]`.

## How it works

**Header checks, outermost first.** Every read is preceded by a bounds check against the
mbuf data length.
1. Ethernet: EtherType must be IPv4.
2. IPv4: version 4; header length from IHL (not assumed to be 20); total length within the
   frame; not a fragment (MF flag and fragment offset both zero); header checksum valid.
3. UDP: protocol 17; destination port 5000, otherwise counted as `drop_other_port`; UDP
   length within the IP datagram.
4. Our 22-byte header: read field by field with `memcpy` and `rte_be_to_cpu_16/32` (the
   payload has no alignment guarantee and the fields are big-endian). Magic, frame size,
   `nseg`, `seq < nseg`, `offset == seq * 1400`, and `length` equal to the payload size
   after the header are all checked.

**Reassembly in any order.** Each chunk is copied to `offset` in a per-frame buffer, so
arrival order does not matter. A per-frame array `got[seq]` records which chunks have
arrived: a second copy of a chunk is counted as a duplicate and not copied.

**Every mbuf is freed**, accepted or dropped. The pool has 8,191 mbufs; leaking dropped
packets would stall reception after about 8,000 packets without any error.

**Knowing when the input is finished.** The pcap port does not report end of file; after
the last packet `rte_eth_rx_burst` simply keeps returning 0. The receiver stops when either
- `--expect N` frames are complete (fastest, when the sender's frame count is known), or
- no packet has arrived for `--idle-ms` (default 200 ms) after the first packet.
  Timing uses `rte_get_timer_cycles()` / `rte_get_timer_hz()`.

  It also gives up if nothing arrives within 10 s. The trade-off: on a real network a
  short idle timeout could cut off a slow tail of packets, and a long one only costs
  waiting time. For a PCAP replay 200 ms is far longer than any gap between packets.

**Output.** Complete frames are written in frame_id order, headerless, 150,528 bytes each:
the same layout as Part 3's `original.bin`, so `cmp` checks the result and
`part5_classify/frame_io.py` reads it directly. `OUTPUT.ids.json` lists the frame IDs
written. An incomplete frame is reported with its missing sequence numbers and left out,
not written with holes; Part 5 can still match the remaining frames by passing the
`.ids.json` as the ID sidecar. The exit status is non-zero if any frame is incomplete.

## Results

Run log: [`results/part4/run-20261024T200549Z.log`](../results/part4/run-20261024T200549Z.log)
(Docker, Ubuntu 22.04, DPDK 21.11.9 from apt, the same packages the instance installs).

| PCAP | Packets | Accepted | Duplicates | Other port | Complete frames | vs `original.bin` |
|---|---:|---:|---:|---:|---:|---|
| in order | 1,728 | 1,728 | 0 | 0 | 16/16 | identical |
| shuffled | 1,728 | 1,728 | 0 | 0 | 16/16 | identical |
| shuffled + 50 duplicates + 20 other-port | 1,798 | 1,728 | 50 | 20 | 16/16 | identical |
| shuffled, 3 chunks of frame 5 removed | 1,725 | 1,725 | 0 | 0 | 15/16 | frame 5 reported missing seq 0, 50, 107 and left out |

Also checked: `part5_classify/frame_io.load_frames` reads the stress-case output with zero
differing pixels against the originals, and reads the packet-loss output with its
`.ids.json` as frames 0-4 and 6-15. With `--expect 16` the receiver stops as soon as the
16th frame completes.

The PCAP replay takes 1-2 ms per run here, but that measures the pcap driver reading a file,
not network throughput, so it is not reported as a performance result.
