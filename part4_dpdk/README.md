# Part 4: Receiving Packets with DPDK

Owner: Fengmao Xie

PDF Steps 4.1-4.2: `dpdk_rx.c` reads the PCAP through the DPDK pcap driver, checks the Ethernet/IPv4/UDP headers, rebuilds each image in any arrival order, ignores duplicates, and writes the complete images to a file for Part 5.
