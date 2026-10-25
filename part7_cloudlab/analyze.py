#!/usr/bin/env python3
"""Part 7: turn the GPUNetIO and perftest logs into CSV tables and two plots.

    python3 part7_cloudlab/analyze.py            # reads results/part7/, writes results/part7/analysis/

Latency: GPUNetIO gpunetio_verbs_write_lat "Unidirectional[usec]" vs perftest ib_write_lat
"t_typical[usec]" (both one-way). Bandwidth: GPUNetIO put_bw (3 runs) and write_bw vs perftest
ib_write_bw "BW average[Gb/sec]". The client logs carry the per-size tables.
"""
import csv
import os
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "part7")
N1 = os.path.join(ROOT, "node1")
OUT = os.path.join(ROOT, "analysis")
NUM = re.compile(r"\d+(?:\.\d+)?")


def table(path):
    """Rows after the '#bytes' header: {bytes: [numbers...]} (trailing perftest noise dropped)."""
    rows, ncol, active = {}, 0, False
    for line in open(path, errors="replace"):
        if "#bytes" in line:
            ncol = len(re.findall(r"#?\w[\w ]*?\[[^\]]*\]|#\w+", line))
            active = True
            continue
        if not active:
            continue
        nums = NUM.findall(line)
        if len(nums) >= 3 and line.strip().split()[0].isdigit():
            rows[int(nums[0])] = [float(x) for x in nums[1:ncol]]
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    gpu_lat = table(os.path.join(N1, "lat_client.txt"))            # iters, uni, bi, kernel
    gpu_lat_p2 = table(os.path.join(N1, "lat_p2_sudo_client.txt")) # -p 2 (GPU rings doorbell), sudo
    put_p2 = table(os.path.join(N1, "put_bw_p2_sudo_client.txt"))
    cpu_lat = table(os.path.join(N1, "ib_lat_all_client.txt"))     # iters, min, max, typical, ...
    put = [table(os.path.join(N1, f)) for f in
           ("put_bw_client.txt", "put_bw_run2_client.txt", "put_bw_run3_client.txt")]
    wbw = table(os.path.join(N1, "write_bw_client.txt"))
    cpu_bw = table(os.path.join(N1, "ib_bw_all_client.txt"))       # iters, peak, avg, msgrate

    with open(os.path.join(OUT, "latency.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bytes", "gpunetio_p1_unidirectional_us", "gpunetio_p2_sudo_unidirectional_us",
                    "perftest_typical_us"])
        for b in sorted(set(gpu_lat) | set(cpu_lat)):
            w.writerow([b, gpu_lat.get(b, [None, None])[1], gpu_lat_p2.get(b, [None, None])[1],
                        cpu_lat.get(b, [None] * 4)[3]])

    with open(os.path.join(OUT, "bandwidth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bytes", "put_bw_run1_gbps", "put_bw_run2_gbps", "put_bw_run3_gbps",
                    "put_bw_mean_gbps", "put_bw_msgrate_mpps_run1", "put_bw_p2_sudo_gbps",
                    "put_bw_p2_sudo_msgrate_mpps", "write_bw_gbps", "perftest_avg_gbps"])
        for b in sorted(set(put[0]) | set(cpu_bw)):
            runs = [r[b][1] for r in put if b in r]
            w.writerow([b] + [r[b][1] if b in r else None for r in put] +
                       [round(statistics.mean(runs), 3) if runs else None,
                        put[0][b][2] if b in put[0] else None,
                        put_p2[b][1] if b in put_p2 else None,
                        put_p2[b][2] if b in put_p2 else None,
                        wbw[b][1] if b in wbw else None,
                        cpu_bw[b][2] if b in cpu_bw else None])

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = sorted(gpu_lat)
    ax.plot(xs, [gpu_lat[b][1] for b in xs], "o-", label="GPUNetIO write_lat, -p 1 (CPU proxy rings doorbell)")
    xs = sorted(gpu_lat_p2)
    ax.plot(xs, [gpu_lat_p2[b][1] for b in xs], "d--", label="GPUNetIO write_lat, -p 2 (GPU rings doorbell, sudo)")
    xs = sorted(cpu_lat)
    ax.plot(xs, [cpu_lat[b][3] for b in xs], "s-", label="perftest ib_write_lat (CPU-initiated)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("one-way latency (us)")
    ax.set_title("RDMA write latency, A30 + ConnectX-6, d7525 (RoCE v2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = sorted(put[0])
    lo = [min(r[b][1] for r in put) for b in xs]
    hi = [max(r[b][1] for r in put) for b in xs]
    mean = [statistics.mean(r[b][1] for r in put) for b in xs]
    ax.fill_between(xs, lo, hi, alpha=0.25)
    ax.plot(xs, mean, "o-", label="GPUNetIO put_bw -p 1 (mean of 3 runs, band = min-max)")
    xs2 = sorted(put_p2)
    ax.plot(xs2, [put_p2[b][1] for b in xs2], "d:", label="GPUNetIO put_bw -p 2 (GPU doorbell, sudo, 1 run)")
    xs = sorted(wbw)
    ax.plot(xs, [wbw[b][1] for b in xs], "^--", label="GPUNetIO write_bw -p 1 (1 run)")
    xs = sorted(cpu_bw)
    ax.plot(xs, [cpu_bw[b][2] for b in xs], "s-", label="perftest ib_write_bw avg (CPU-initiated)")
    ax.axhline(200, color="grey", lw=0.8, ls=":")
    ax.text(2, 188, "200 Gb/s link rate", color="grey", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bandwidth (Gb/s)")
    ax.set_title("RDMA write bandwidth, A30 + ConnectX-6, d7525 (RoCE v2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bandwidth.png"), dpi=150)

    b64 = 65536
    print("smallest-message one-way latency: GPUNetIO %.2f us (%d B), perftest %.2f us (%d B)" % (
        gpu_lat[min(gpu_lat)][1], min(gpu_lat), cpu_lat[min(cpu_lat)][3], min(cpu_lat)))
    print("-p 2 (sudo) smallest-message latency: %.2f us; put_bw -p 2 64 KB %.2f Gb/s" % (
        gpu_lat_p2[min(gpu_lat_p2)][1], put_p2[b64][1]))
    print("64 KB bandwidth: put_bw runs %s Gb/s, write_bw %.2f, perftest avg %.2f Gb/s" % (
        [r[b64][1] for r in put], wbw[b64][1], cpu_bw[b64][2]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
